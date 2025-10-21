import os
import pickle
import hashlib
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gamma, norm, wishart, multivariate_normal
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp, digamma
from scipy.optimize import minimize

from dipy.reconst.dti import mean_diffusivity, fractional_anisotropy
from dipy.io.image import load_nifti, save_nifti  # for loading / saving imaging datasets
from dipy.io.gradients import read_bvals_bvecs  # for loading / saving our bvals and bvecs
from dipy.core.gradients import gradient_table  # for constructing gradient table from bvals/bvecs
from dipy.data import get_fnames  # for small datasets that we use in tests and examples
from dipy.segment.mask import median_otsu  # for masking out the background
import dipy.reconst.dti as dti  # for diffusion tensor model fitting and metrics

import pandas as pd


def disk_memoize(cache_dir="cache"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            force = kwargs.pop("force_recompute", False)

            os.makedirs(cache_dir, exist_ok=True)

            func_name = func.__name__
            key = (func_name, args, kwargs)
            hash_str = hashlib.md5(pickle.dumps(key)).hexdigest()
            cache_path = os.path.join(cache_dir, f"{func_name}_{hash_str}.pkl")

            if not force and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator


@disk_memoize()
def get_preprocessed_data():
    # Load the masked data, background mask, and gradient information
    data, mask, gtab = get_data()

    # Initialize a diffusion tensor model (DTI) with S0 estimation enabled
    tenmodel = dti.TensorModel(gtab, return_S0_hat=True)

    # Extract the signal for a single voxel (coordinates chosen for this project)
    y = data[35, 35, 30, :]

    # Fit the DTI model to this voxel's signal
    tenfit = tenmodel.fit(y)

    # Extract point estimates: baseline signal, eigenvalues, and eigenvectors
    S0 = tenfit.S0_hat
    evals = tenfit.evals
    evecs = tenfit.evecs
    point_estimate = [S0, evals, evecs]

    # Return the raw voxel signal, point estimate, and gradient table
    return y, point_estimate, gtab


def get_data():
    # Download filenames for the Stanford HARDI dataset if not already cached
    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

    # Load the raw 4D dataset: dimensions are (x, y, z, diffusion measurements)
    data, _ = load_nifti(hardi_fname)

    # Read diffusion weighting information (b-values, b-vectors) and build gradient table
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    # Apply brain masking and cropping to remove background and save compute
    maskdata, mask = median_otsu(
        data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True, dilate=2
    )

    # Print the final data shape for confirmation
    print('Loaded data with shape: (%d, %d, %d, %d)' % maskdata.shape)

    return maskdata, mask, gtab


def compute_D(evals, V):
    # Ensure inputs have the correct batch dimensions
    if evals.ndim == 1:
        evals = evals[None, None, :]
    elif evals.ndim == 2:
        evals = evals[:, None, :]
    if V.ndim == 2:
        V = V[None, :, :]

    # Compute D = V Λ V.T as V (V @ Λ).T
    V_scaled = V * evals
    D = np.matmul(V, np.transpose(V_scaled, axes=[0, 2, 1]))

    return D


def theta_from_D(D):
    # Compute Cholesky factor (lower-triangular L) of D
    L = np.linalg.cholesky(D)

    # Indices of lower-triangular entries (including diagonal)
    p = D.shape[0]
    tril_indices = np.tril_indices(p)
    theta = []

    # Store log of diagonal entries, raw off-diagonal entries
    for i, j in zip(*tril_indices):
        if i == j:
            theta.append(np.log(L[i, j]))  # Diagonal: log-transform
        else:
            theta.append(L[i, j])  # Off-diagonal: raw value

    return np.array(theta)


def D_from_theta(theta):
    # Ensure theta is an array and check shape
    theta = np.asarray(theta)
    *batch_shape, _ = theta.shape
    assert theta.shape[-1] == 6, "Last dimension must be 6 for 3x3 lower-triangular matrices."

    # Initialize lower-triangular matrix L
    L = np.zeros((*batch_shape, 3, 3), dtype=theta.dtype)

    # Fill L with exponentiated diagonals and raw off-diagonals
    tril_indices = np.tril_indices(3)
    for k, (i, j) in enumerate(zip(*tril_indices)):
        if i == j:
            L[..., i, j] = np.exp(theta[..., k])  # Diagonal
        else:
            L[..., i, j] = theta[..., k]  # Off-diagonal

    # Reconstruct D = L @ L.T (batch-aware matrix multiplication)
    D = L @ np.swapaxes(L, -1, -2)

    return D.squeeze()


def grad_D_wrt_theta_at_D(D):
    # Get Cholesky factor of D and set up indices for lower-triangular entries
    p = D.shape[0]
    L = np.linalg.cholesky(D)
    tril_indices = np.tril_indices(p)
    num_params = len(tril_indices[0])

    # Prepare output container
    grad_D = np.zeros((p, p, num_params))

    # Loop over all parameters in theta
    for k, (m, n) in enumerate(zip(*tril_indices)):

        # Build a basis matrix for the effect of this parameter
        E_mn = np.zeros((p, p))
        if m == n:
            # Diagonal: dL_mm/dtheta = L_mm since L_mm = exp(theta)
            factor = L[m, n]
        else:
            # Off-diagonal: dL_mn/dtheta = 1
            factor = 1.0
        E_mn[m, n] = factor

        # Work out the corresponding change in D
        dD_k = E_mn @ L.T + L @ E_mn.T
        grad_D[:, :, k] = dD_k

    return grad_D


class frozen_prior:
    """Gamma priors for S0 and eigenvalues; uniform on SO(3) for V (constant)."""

    def __init__(self, alpha_S=2.0, theta_S=500.0, alpha_lam=4.0, theta_lam=2.5e-4):
        self.alpha_S = float(alpha_S)
        self.theta_S = float(theta_S)
        self.alpha_lam = float(alpha_lam)
        self.theta_lam = float(theta_lam)

    def rvs(self, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        S0 = gamma(a=self.alpha_S, scale=self.theta_S).rvs(random_state=rng)
        lams = gamma(a=self.alpha_lam, scale=self.theta_lam).rvs(size=3, random_state=rng)
        V = Rotation.random(random_state=rng).as_matrix()
        D = compute_D(lams, V).squeeze()
        return S0, D

    def logpdf(self, S0, D):
        # log p(S0) + sum_j log p(lambda_j); V is uniform -> constant we ignore
        lp = gamma(a=self.alpha_S, scale=self.theta_S).logpdf(S0)
        evals, _ = np.linalg.eigh(D)
        evals = np.sort(evals)[::-1]  # descending (λ1 ≥ λ2 ≥ λ3)
        lp += np.sum(gamma(a=self.alpha_lam, scale=self.theta_lam).logpdf(evals))
        return float(lp)


class frozen_likelihood:
    """Gaussian noise: y_i ~ N(S0 * exp(-q_i^T D q_i), sigma^2)."""

    def __init__(self, y, gtab, sigma=29.0):
        self.y = np.asarray(y, dtype=float)
        self.sigma = float(sigma)
        self.gtab = gtab
        # q encodes gradient direction & strength: q_i = sqrt(b_i) * bvec_i
        self.q = np.sqrt(self.gtab.bvals)[:, None] * self.gtab.bvecs  # (N,3)

    def _signal(self, S0, D):
        # S(x) = S0 * exp(-x^T D x), here x = q_i
        quad = np.einsum('ni,ij,nj->n', self.q, D, self.q)
        return S0 * np.exp(-quad)

    def logpdf(self, S0, D):
        s = self._signal(S0, D)
        resid = self.y - s
        n = self.y.size
        ll = -0.5 * n * np.log(2.0 * np.pi * self.sigma ** 2) - 0.5 * np.sum(resid ** 2) / (self.sigma ** 2)
        return float(ll)


y, point_estimate, gtab = get_preprocessed_data()
S0, evals, evecs = point_estimate
D = compute_D(evals, evecs).squeeze()
prior = frozen_prior(2.0, 500.0, 4.0, 2.5e-4)
print(round(prior.logpdf(S0, D), 3))  # <-- Q1 answer
like = frozen_likelihood(y, gtab, sigma=29)
print(round(like.logpdf(S0, D), 3))  # <-- Q2 answer


def _S0_D_to_theta(S0, D):
    L = np.linalg.cholesky(D)
    t0 = np.log(S0)
    t11 = np.log(L[0, 0])
    t21 = L[1, 0]
    t22 = np.log(L[1, 1])
    t31 = L[2, 0]
    t32 = L[2, 1]
    t33 = np.log(L[2, 2])
    return np.array([t0, t11, t21, t22, t31, t32, t33], dtype=float)


def _theta_to_S0_D(theta):
    t0, t11, t21, t22, t31, t32, t33 = theta
    S0 = float(np.exp(t0))
    L = np.array([
        [np.exp(t11), 0.0, 0.0],
        [t21, np.exp(t22), 0.0],
        [t31, t32, np.exp(t33)]
    ], dtype=float)
    D = L @ L.T
    w, V = np.linalg.eigh(D)
    D = (V * np.clip(w, 1e-12, None)) @ V.T
    return S0, D


class variational_posterior:
    """
    Mean-field Gaussian q(theta) = N(mu, diag(sigma^2)) over 7-dim theta:
    [t0, t11, t21, t22, t31, t32, t33]. sigma = softplus(rho).
    """

    def __init__(self, mu, rho):
        self.mu = np.asarray(mu, dtype=float)
        self.rho = np.asarray(rho, dtype=float)

    @staticmethod
    def _softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.) + 1e-6

    def sigma(self):
        return self._softplus(self.rho)

    def logpdf_theta(self, theta):
        theta = np.atleast_2d(theta)
        mu = self.mu[None, :]
        sig = self.sigma()[None, :]
        z = (theta - mu) / sig
        return -0.5 * np.sum(z ** 2 + 2.0 * np.log(sig) + np.log(2 * np.pi), axis=1)

    def rvs_theta(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        eps = rng.standard_normal((size, self.mu.size))
        return self.mu[None, :] + eps * self.sigma()[None, :]

    def rvs(self, size=1, random_state=None):
        thetas = self.rvs_theta(size=size, random_state=random_state)
        S0s, evals_list, evecs_list = [], [], []
        for th in thetas:
            S0, D = _theta_to_S0_D(th)
            lams, V = np.linalg.eigh(D)
            idx = np.argsort(lams)  # ascending
            S0s.append(S0)
            evals_list.append(lams[idx])
            evecs_list.append(V[:, idx])
        return np.array(S0s), np.array(evals_list), np.array(evecs_list)




# BAYESIAN OPTIMIZATION OF IS HYPERPARAMETERS
class IS_hyperparam_bayesian_optimization:
    def __init__(self, y, gtab, S0_init, D_init):
        self.y = y
        self.gtab = gtab
        self.S0_init = S0_init
        self.D_init = D_init

    def objective_function_ess(self, logweights):
        return np.exp(-logsumexp(2 * logweights))

    def acquisition_function(self, best_ess_value, mean, std):
        # Expected improvement acquisition function

        # preventing division by 0
        if std < 10 ** -6:
            std = 10 ** -6

        cdf = norm.cdf(x=(mean - best_ess_value) / std)
        pdf = norm.pdf(x=(mean - best_ess_value) / std)

        return (mean - best_ess_value) * cdf + std * pdf

    def kernel(self, reference_data, new_data, std_kernel=2):
        # Function calculates a Gaussian kernel
        gamma_vec_ref = reference_data["gamma"].to_numpy()
        nu_vec_ref = reference_data["nu"].to_numpy()

        gamma_vec_new = new_data["gamma"].to_numpy()
        nu_vec_new = new_data["nu"].to_numpy()

        num_ref_data = len(gamma_vec_ref)
        num_new_data = len(gamma_vec_new)

        # Initialize kernel matrix
        kernel_mat = np.zeros([num_ref_data, num_new_data])
        denominator = 2 * std_kernel ** 2
        for i in range(num_ref_data):
            ref_point = np.array([gamma_vec_ref[i], nu_vec_ref[i]])
            for k in range(num_new_data):
                new_point = np.array([gamma_vec_new[k], nu_vec_new[k]])
                dist = np.linalg.norm(ref_point - new_point)
                kernel_mat[i, k] = np.exp(-np.square(dist) / denominator)

        return kernel_mat

    def gaussian_process_mean_std(self, training_data, predicted_points):
        # Calculates the new mean and std of the gaussian process, given old data and new, predicted data points.
        ess_training = training_data["ess"].to_numpy().T

        cov_inv = np.linalg.inv(self.kernel(training_data, training_data))
        cross_cov = self.kernel(training_data, predicted_points)
        self_cov = self.kernel(predicted_points, predicted_points)

        new_mean = cross_cov.T @ cov_inv @ ess_training
        new_covariance = self_cov - cross_cov.T @ cov_inv @ cross_cov
        new_variance = np.maximum(0, np.diag(new_covariance).flatten())

        return new_mean, np.sqrt(new_variance)

    def optimize(self,
                 num_iterations: int = 200,
                 num_gamma_samples: int = 50,
                 num_initial_samples: int = 5,
                 nu_max: int = 10,
                 nu_min: int = 3
                 ):

        # Initial samples
        gamma_initial_vec = np.random.uniform(0, 10, size=num_initial_samples)
        nu_initial_vec = np.random.uniform(nu_min, nu_max, size=num_initial_samples).astype(int)
        ess_gamma_nu_list = []
        for i, nu_sample in enumerate(nu_initial_vec):
            gamma_sample = gamma_initial_vec[i]
            logweights, _, _, _ = importance_sampling(
                y=self.y,
                gtab=self.gtab,
                S0_init=self.S0_init,
                D_init=self.D_init,
                n_samples=10000,
                IS_gamma_param=gamma_sample,
                IS_nu_param=nu_sample,
                enable_logweights=True
            )

            ess = self.objective_function_ess(logweights=logweights)
            ess_gamma_nu_list.append({"ess": ess, "gamma": gamma_sample, "nu": nu_sample})

        evaluated_data = pd.DataFrame(ess_gamma_nu_list)

        nu_vec = np.arange(nu_min, nu_max + 1)
        for i in range(num_iterations):
            gamma_sample_vec = np.random.exponential(scale=1.5, size=num_gamma_samples)

            gen_data_list = []
            for nu_sample in nu_vec:
                for gamma_sample in gamma_sample_vec:
                    gen_data_list.append({"gamma": gamma_sample, "nu": nu_sample})

            generated_data = pd.DataFrame(gen_data_list)

            mean_vec, std_vec = self.gaussian_process_mean_std(evaluated_data, generated_data)

            highest_ess_val = evaluated_data["ess"].max()
            ei_vec = np.zeros(len(mean_vec))
            for i, mean in enumerate(mean_vec):
                std = std_vec[i]
                ei_vec[i] = self.acquisition_function(highest_ess_val, mean, std)

            highest_ei_index = np.argmax(ei_vec)
            new_gamma = generated_data.loc[highest_ei_index, "gamma"]
            new_nu = generated_data.loc[highest_ei_index, "nu"]

            logweights, _, _, _ = importance_sampling(
                y=self.y,
                gtab=self.gtab,
                S0_init=self.S0_init,
                D_init=self.D_init,
                n_samples=10000,
                IS_gamma_param=new_gamma,
                IS_nu_param=new_nu,
                enable_logweights=True
            )
            ess = self.objective_function_ess(logweights=logweights)
            evaluated_data = pd.concat(
                [evaluated_data, pd.DataFrame({"ess": [ess], "gamma": [new_gamma], "nu": [new_nu]})],
                ignore_index=True)

        highest_ess_index = evaluated_data["ess"].idxmax()
        highest_ess_value = evaluated_data.loc[highest_ess_index, "ess"]
        best_gamma = evaluated_data.loc[highest_ess_index, "gamma"]
        best_nu = evaluated_data.loc[highest_ess_index, "nu"]

        return best_gamma, best_nu, highest_ess_value, evaluated_data



"""
=============================================================================
Inference Methods (need to be implemented)
=============================================================================
Students: implement one method each (MH, IS, VI, or Laplace).
Uses memoization to speed up repeated runs.
"""

@disk_memoize()
def metropolis_hastings(n_samples, y, gtab, S0_init, D_init, gamma_prop_param, nu, plot_traces=False):
    """

    """
    # Initialize prior and likelihood models
    prior = frozen_prior()
    likelihood = frozen_likelihood(gtab, y)

    # --- Initial state ---
    current_S0 = S0_init
    current_D = D_init
    current_evals, current_evecs = np.linalg.eigh(current_D)

    # Sort eigenvalues/eigenvectors and reconstruct D for consistency
    idx = np.argsort(current_evals)[::-1]
    current_evals = current_evals[idx]
    current_evecs = current_evecs[:, idx]
    current_D = compute_D(current_evals, current_evecs).squeeze()

    # Calculate initial log-posterior
    current_log_prior = prior.logpdf(current_S0, current_evals, current_evecs)
    current_log_likelihood = likelihood.logpdf(current_S0, current_evecs, current_evals)
    current_log_posterior = current_log_prior + current_log_likelihood

    # --- Storage for samples ---
    S0_samples = np.zeros(n_samples)
    evals_samples = np.zeros((n_samples, 3))
    evecs_samples = np.zeros((n_samples, 3, 3))

    # --- Acceptance counters ---
    accept_S0, accept_D = 0, 0

    # --- Gamma proposal parameters for S0 ---
    # Proposal q(S0'|S0) = Gamma(alpha, theta) where E[S0'] = alpha * theta = S0
    gamma_sq = gamma_prop_param ** 2
    alpha_prop = 1.0 / gamma_sq

    print("Starting Metropolis-Hastings sampling with Gamma-Wishart proposals...")

    for i in range(n_samples):
        if i > 0 and i % 1000 == 0:
            print(f"Sample {i}/{n_samples}, Acceptance rates: "
                  f"S0: {accept_S0 / i:.3f}, D: {accept_D / i:.3f}")

        # Store current state (to avoid overwriting if one block rejects)
        # Note: We do *not* store here, as the sampling process is sequential.
        # The stored value should be the one after *both* potential updates.

        # =====================================================================
        # === Block 1: Update S0 using Gamma proposal (given D) ===
        # =====================================================================

        # 1. Propose S0' from q(S0'|S0) = Gamma(alpha_prop, theta_prop)
        theta_prop = current_S0 / alpha_prop  # such that E[S0'] = current_S0
        proposed_S0 = gamma.rvs(a=alpha_prop, scale=theta_prop)

        # Ensure S0' is positive (should be guaranteed by Gamma, but check anyway)
        if proposed_S0 > 0:
            # 2. Calculate log-posterior for the proposed state (S0', D)
            proposed_log_prior_S0 = prior.logpdf(proposed_S0, current_evals, current_evecs)
            proposed_log_likelihood_S0 = likelihood.logpdf(proposed_S0, current_evecs, current_evals)
            proposed_log_posterior_S0 = proposed_log_prior_S0 + proposed_log_likelihood_S0

            # 3. Calculate log Hastings correction term: log(q(S0|S0') / q(S0'|S0))
            # Forward proposal density: q(S0'|S0)
            log_q_fwd_S0 = gamma.logpdf(proposed_S0, a=alpha_prop, scale=theta_prop)

            # Reverse proposal density: q(S0|S0')
            theta_rev = proposed_S0 / alpha_prop  # Scale for the reverse proposal
            log_q_rev_S0 = gamma.logpdf(current_S0, a=alpha_prop, scale=theta_rev)

            log_hastings_S0 = log_q_rev_S0 - log_q_fwd_S0

            # 4. Calculate log-acceptance ratio
            # log(alpha) = log(P(S0') + L(S0') - P(S0) - L(S0) + log_hastings)
            # Use current_log_posterior (full P+L) for rejection to be accurate
            log_alpha_S0 = (
                    proposed_log_posterior_S0 - current_log_posterior + log_hastings_S0
            )

            # 5. Acceptance step
            if np.log(np.random.rand()) < log_alpha_S0:
                current_S0 = proposed_S0
                current_log_posterior = proposed_log_posterior_S0  # Update the full log posterior
                accept_S0 += 1

        try:

            proposed_D = wishart.rvs(df=nu, scale=current_D / nu)  # Use current_D/nu for E[D'] = current_D

            # Check for positive definite / valid D before continuing
            np.linalg.cholesky(proposed_D)

        except np.linalg.LinAlgError:
            # Proposed D is not positive definite, or Wishart failed
            log_alpha_D = -np.inf  # Reject
            proposed_log_posterior_D = -np.inf
        except ValueError:  # scipy.stats.wishart.logpdf can raise this if scale is bad
            log_alpha_D = -np.inf
            proposed_log_posterior_D = -np.inf
        else:
            # Proposed D is valid:
            # 2. Decompose and sort D'
            proposed_evals, proposed_evecs = np.linalg.eigh(proposed_D)
            idx_D = np.argsort(proposed_evals)[::-1]

            new_idx = np.array([idx_D[2], idx_D[1], idx_D[0]])  # 索引 0 对应 lambda2，索引 2 对应 lambda0

            # 5. 重新排序特征值 (只是为了保持 evals/evecs 数组内部的一致性，但现在是升序)
            proposed_evals = proposed_evals[new_idx]

            proposed_evecs = proposed_evecs[:, new_idx]

            # proposed_evals = proposed_evals[idx_D]
            # proposed_evecs = proposed_evecs[:, idx_D]

            # 3. Calculate log-posterior for the proposed state (S0, D')
            proposed_log_prior_D = prior.logpdf(current_S0, proposed_evals, proposed_evecs)
            proposed_log_likelihood_D = likelihood.logpdf(current_S0, proposed_evecs, proposed_evals)
            proposed_log_posterior_D = proposed_log_prior_D + proposed_log_likelihood_D

            # 4. Calculate log Hastings correction term: log(q(D|D') / q(D'|D))
            # Note: Wishart logpdf requires a positive definite scale matrix.
            Sigma_fwd = current_D / nu
            Sigma_rev = proposed_D / nu

            try:
                # Forward proposal density: q(D'|D)
                log_q_fwd_D = wishart.logpdf(proposed_D, df=nu, scale=Sigma_fwd)
                # Reverse proposal density: q(D|D')
                log_q_rev_D = wishart.logpdf(current_D, df=nu, scale=Sigma_rev)

                log_hastings_D = log_q_rev_D - log_q_fwd_D

                # 5. Calculate log-acceptance ratio
                # log(alpha) = log(P(D') + L(D') - P(D) - L(D) + log_hastings)
                log_alpha_D = (
                        proposed_log_posterior_D - current_log_posterior + log_hastings_D
                )
            except np.linalg.LinAlgError:
                # Can fail if scale matrix is not positive definite (e.g., if D' had a tiny negative eigenvalue due to numerical error)
                log_alpha_D = -np.inf
            except ValueError:
                # Catch other potential errors in logpdf calculation
                log_alpha_D = -np.inf

        # 6. Acceptance step
        if np.log(np.random.rand()) < log_alpha_D:
            current_D = proposed_D  # Keep the tensor (even though we'll recompute it from evals/evecs)
            current_evals = proposed_evals  # Store sorted evals
            current_evecs = proposed_evecs  # Store sorted evecs
            current_log_posterior = proposed_log_posterior_D  # Update the full log posterior
            accept_D += 1

        # Store the current state after both updates
        S0_samples[i] = current_S0
        evals_samples[i, :] = current_evals
        evecs_samples[i, :, :] = current_evecs

    print("\n--- Final Acceptance Rates ---")
    print(f"S0 (Gamma proposal): {accept_S0 / n_samples:.3f}")
    print(f"D (Wishart proposal): {accept_D / n_samples:.3f}")

    return S0_samples, evals_samples, evecs_samples




@disk_memoize()
def importance_sampling(
        y,
        gtab,
        S0_init,
        D_init,
        n_samples: int,
        IS_gamma_param: float,
        IS_nu_param: int,
        enable_logweights: bool = False,
        enable_sequential_mc: bool = False,
        n_sequential_mc_iterations: int = 50,
        n_kernel_iterations: int = 10,
        smc_evals_std=0.0002,
        smc_rot_std=2,
        smc_gamma_param=0.02
):
    def IS_algorithm():
        # Generating samples
        S0_samples = gamma.rvs(a=IS_gamma_param ** -2, scale=(IS_gamma_param ** 2) * S0_init, size=n_samples)
        log_qS0 = gamma.logpdf(x=S0_samples, a=IS_gamma_param ** -2, scale=(IS_gamma_param ** 2) * S0_init)
        D_samples = wishart.rvs(df=IS_nu_param, scale=IS_nu_param * D_init, size=n_samples)

        evals_samples, evecs_samples = np.linalg.eigh(D_samples)

        # Defining the prior and likelihood functions
        likelihood = frozen_likelihood(gtab=gtab, y=y)
        prior = frozen_prior()

        # Iterating over each sample
        log_qD = np.zeros(n_samples)
        for i in range(n_samples):
            log_qD[i] = wishart.logpdf(x=D_samples[i, :, :], df=IS_nu_param, scale=D_init)

        log_q = log_qS0 + log_qD
        logpdf_likelihood = likelihood.logpdf(S0=S0_samples, evecs=evecs_samples, evals=evals_samples)
        logpdf_prior = prior.logpdf(S0=S0_samples, evals=evals_samples, evecs=evecs_samples)

        log_weights = logpdf_prior + logpdf_likelihood - log_q
        importance_weights_normalized = log_weights - logsumexp(log_weights)

        if enable_logweights == True:
            importance_weights = importance_weights_normalized

        elif enable_logweights == False:
            importance_weights = np.exp(importance_weights_normalized)

        return importance_weights, S0_samples, evals_samples, evecs_samples

    def SMC_algorithm():
        def effective_sample_size(logweights):
            # Defining the metric
            ess = np.exp(-logsumexp(2 * logweights))
            return ess

        def resample(logweights, S0_samples, evals_samples, evecs_samples):
            # Resamples [multinomial] the particles according to their weights (normalized weights)
            indices = np.random.choice(a=n_samples, size=n_samples, p=np.exp(logweights))

            S0_resampled = S0_samples[indices]
            evals_resampled = evals_samples[indices]
            evecs_resampled = evecs_samples[indices]

            return S0_resampled, evals_resampled, evecs_resampled

        def evals_norm_rvs(evals_means, std):
            # Evals proposal for rejuvenation. Assumes gaussian dist noise around each separate value in the input evals
            evals_sample = norm.rvs(loc=evals_means, scale=std, size=3)

            # All evals need to be positive for positive definite matrix
            if np.all(evals_sample > 0):
                return evals_sample
            else:
                while np.all(evals_sample > 0) == False:
                    index = np.where(evals_sample <= 0)[0]
                    evals_sample[index] = norm.rvs(loc=evals_means[index], scale=std, size=len(index))
                return evals_sample

        def evecs_rotation_rvs(evecs_mean, std):
            # Returns evecs randomly rotated (normal dist) in yaw, roll and pitch around itself with std in [degrees].
            std_rad = std * 2 * np.pi / 360
            rotation_angles = norm.rvs(loc=0, scale=std_rad, size=3)
            rotation_matrix = Rotation.from_rotvec(rotation_angles).as_matrix()
            evecs = evecs_mean @ rotation_matrix
            if np.linalg.det(evecs) < 0:
                evecs[2] = -evecs[2]
            return evecs

        def evec_rotation_logpdf(x, loc, std):
            # Returns the logpdf of rotation around yaw, pitch and roll given an std in degrees (assuming gaussian dist)
            std_rad = std * 2 * np.pi / 360
            x_loc_rotation_matrix = loc.T @ x
            angles_difference = Rotation.from_matrix(x_loc_rotation_matrix).as_rotvec()

            return multivariate_normal.logpdf(x=angles_difference, cov=(std_rad ** 2) * np.eye(3))

        def markov_kernel_rejuvination(S0_samples,
                                       evals_samples,
                                       evecs_samples,
                                       phi
                                       ):
            accepted_particles = 0
            proposed_particles = 0

            # Rejuvination of samples using Metropolis-Hastings (same proposal is MH algorithm)
            prior = frozen_prior()
            likelihood = frozen_likelihood(y=y, gtab=gtab)

            for i in range(n_samples):
                # Initializing values for sample.
                S0 = S0_samples[i]
                evals = evals_samples[i]
                evecs = evecs_samples[i]

                for k in range(n_kernel_iterations):
                    proposed_particles += 1

                    # Sampling from proposal
                    S0_prime = gamma.rvs(a=smc_gamma_param ** -2, scale=(smc_gamma_param ** 2) * S0, size=1)
                    evals_prime = evals_norm_rvs(evals, std=smc_evals_std)
                    evecs_prime = evecs_rotation_rvs(evecs, std=smc_rot_std)
                    D_prime = compute_D(evals_prime, evecs_prime)[0]

                    # Calculating the acceptance prob for the proposed samples
                    log_pi_z_prime = (prior.logpdf(S0_prime, D_prime) +
                                      phi * likelihood.logpdf(S0_prime, D_prime)
                                      )

                    log_q_z = (gamma.logpdf(x=S0, a=smc_gamma_param ** -2, scale=(smc_gamma_param ** 2) * S0_prime) +
                               np.sum(norm.logpdf(x=evals, loc=evals_prime, scale=smc_evals_std)) +
                               evec_rotation_logpdf(x=evecs, loc=evecs_prime, std=smc_rot_std)
                               )

                    log_pi_z = prior.logpdf(S0, D_prime) + phi * likelihood.logpdf(S0, D_prime)

                    log_q_z_prime = (
                            gamma.logpdf(x=S0_prime, a=smc_gamma_param ** -2, scale=(smc_gamma_param ** 2) * S0) +
                            np.sum(norm.logpdf(x=evals_prime, loc=evals, scale=smc_evals_std)) +
                            evec_rotation_logpdf(x=evecs_prime, loc=evecs, std=smc_rot_std)
                    )

                    log_acceptance_prob = log_pi_z_prime + log_q_z - log_pi_z - log_q_z_prime

                    if log_acceptance_prob > 0:
                        accepted_particles += 1

                        S0 = S0_prime
                        evals = evals_prime
                        evecs = evecs_prime

                    elif np.random.binomial(1, np.exp(log_acceptance_prob), 1) == 1:
                        accepted_particles += 1

                        S0 = S0_prime
                        evals = evals_prime
                        evecs = evecs_prime

                # Rejuvinated values
                S0_samples[i] = S0
                evals_samples[i] = evals
                evecs_samples[i] = evecs

            acceptance_ratio = accepted_particles / proposed_particles

            return S0_samples, evals_samples, evecs_samples, acceptance_ratio

        # RUNNING THE ALGORITHM ---------------------------------------------------------------
        # Initializing the annealment vector
        phi_vec = np.zeros(n_sequential_mc_iterations)
        S0_samples = np.zeros(n_samples)
        D_samples = np.zeros((n_samples, 3, 3))

        # Initializing by sampling from the prior
        prior = frozen_prior()
        likelihood = frozen_likelihood(y=y, gtab=gtab)

        for i in range(n_samples):
            S0_samples[i], D_samples[i] = prior.rvs(n_samples)

        evals_samples, evecs_samples = np.linalg.eigh(D_samples)

        logweights = np.log(np.ones(n_samples) / n_samples)  # initializing the weights (uniformely)

        acceptance_ratio = 1

        for i in range(1, n_sequential_mc_iterations):
            # Calculating the new annealment exponent
            phi_vec[i] = (i / n_sequential_mc_iterations) ** 3
            delta_phi = phi_vec[i] - phi_vec[i - 1]

            loglikelihood = 0
            for S0_sample, D_sample in zip(S0_samples, D_samples):
                loglikelihood += likelihood.logpdf(S0_sample, D_sample)

            # Updating weights and normalizing
            logweights = logweights + delta_phi * loglikelihood
            logweights = logweights - logsumexp(logweights)

            # Calculate the ess value and resample (and reset weights)
            ess = effective_sample_size(logweights)
            if ess < n_samples / 2:
                S0_samples, evals_samples, evecs_samples = resample(
                    logweights,
                    S0_samples,
                    evals_samples,
                    evecs_samples
                )
                logweights = np.log(np.ones(n_samples) / n_samples)

            # Rejuvinate with markov kernel
            S0_samples, evals_samples, evecs_samples, acceptance_ratio = markov_kernel_rejuvination(
                S0_samples,
                evals_samples,
                evecs_samples,
                phi_vec[i]
            )

            D_samples = compute_D(evals_samples, evecs_samples)

            print(f'\ni = {i}')
            print(f'Acceptance ratio = {100 * acceptance_ratio:.1f}%')
            print(f'Unique S0 samples: {100 * (len(np.unique(np.round(S0_samples, 6))) / len(S0_samples)):.1f}%')

        # No need for resampling on last iteration
        delta_phi = 1 - phi_vec[-1]

        loglikelihood = 0
        for S0_sample, D_sample in zip(S0_samples, D_samples):
            loglikelihood += likelihood.logpdf(S0_sample, D_sample)

        logweights = logweights + delta_phi * loglikelihood
        logweights = logweights - logsumexp(logweights)

        # Returning the particles and weights
        if enable_logweights == True:
            importance_weights = logweights

        elif enable_logweights == False:
            importance_weights = np.exp(logweights)

        for i in range(n_samples):
            indices = np.argsort(evals_samples[i, :])[::-1]
            evals_samples[i, :] = evals_samples[i, indices]
            evecs_samples[i, :, :] = evecs_samples[i, :, indices]

        return importance_weights, S0_samples, evals_samples, evecs_samples

    if enable_sequential_mc == False:
        return IS_algorithm()

    elif enable_sequential_mc == True:
        return SMC_algorithm()


@disk_memoize()
def variational_inference(max_iters=1200, K=24, learning_rate=0.01, print_every=100):
    # --- data & model pieces
    y, point_estimate, gtab = get_preprocessed_data(force_recompute=False)
    S0_init, evals_init, evecs_init = point_estimate
    D_init = compute_D(evals_init, evecs_init).squeeze()
    like = frozen_likelihood(y=y, gtab=gtab, sigma=29.0)
    prior = frozen_prior(alpha_S=2.0, theta_S=500.0, alpha_lam=4.0, theta_lam=2.5e-4)

    # log-joint and its gradient in theta (finite differences, stable in 7D)
    def log_joint_theta(theta):
        S0, D = _theta_to_S0_D(theta)
        return prior.logpdf(S0, D) + like.logpdf(S0, D)

    def grad_log_joint(theta, eps_fd=1e-5):
        g = np.zeros_like(theta)
        f0 = log_joint_theta(theta)
        for i in range(theta.size):
            th = theta.copy();
            th[i] += eps_fd
            g[i] = (log_joint_theta(th) - f0) / eps_fd
        return g

    # --- initialize around DTI; start fairly tight
    theta0 = _S0_D_to_theta(float(S0_init), D_init)
    mu = theta0.copy()
    rho = np.full_like(mu, -3.0)  # softplus(-3) ≈ 0.050  (initial sigma)
    sig = variational_posterior._softplus(rho)

    # Adam state
    def adam_step(p, g, m, v, t, lr=learning_rate, b1=0.9, b2=0.999, eps=1e-8):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat, vhat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        p = p + lr * mhat / (np.sqrt(vhat) + eps)  # ascent
        return p, m, v

    m_mu = np.zeros_like(mu);
    v_mu = np.zeros_like(mu)
    m_rh = np.zeros_like(rho);
    v_rh = np.zeros_like(rho)

    # train
    for t in range(1, max_iters + 1):
        sig = variational_posterior._softplus(rho)
        eps = np.random.randn(K, mu.size)
        thetas = mu[None, :] + sig[None, :] * eps

        # estimate ELBO and gradients
        logp = np.empty(K)
        g_lp = np.empty_like(thetas)
        for i in range(K):
            th = thetas[i]
            logp[i] = log_joint_theta(th)
            g_lp[i] = grad_log_joint(th)

        # entropy of diag normal and its gradient
        H = 0.5 * np.sum(np.log(2 * np.pi * np.e * sig ** 2))
        elbo = float(np.mean(logp) + H)

        # pathwise gradients
        g_mu = np.mean(g_lp, axis=0)  # (7,)
        g_sig = np.mean(g_lp * eps, axis=0) + 1.0 / sig  # (7,)

        # chain rule to rho (sigma = softplus(rho), ds/drho = sigmoid(rho))
        ds_drho = 1.0 / (1.0 + np.exp(-rho))
        g_rho = g_sig * ds_drho

        # gradient clipping (stability)
        def clip(g, c=1.0):
            n = np.linalg.norm(g)
            return g if n <= c else g * (c / (n + 1e-12))

        g_mu = clip(g_mu, 1.0)
        g_rho = clip(g_rho, 1.0)

        # Adam updates
        mu, m_mu, v_mu = adam_step(mu, g_mu, m_mu, v_mu, t)
        rho, m_rh, v_rh = adam_step(rho, g_rho, m_rh, v_rh, t)

        # keep scales reasonable
        rho = np.clip(rho, -6.0, 2.0)  # sigma roughly in [~0.002, ~0.88]

        if t % print_every == 0 or t == 1:
            print(f"[VI-reparam] iter {t:4d}: ELBO≈{elbo:.3f}; "
                  f"||mu||={np.linalg.norm(mu):.3f}, mean(sigma)={np.mean(sig):.3f}")

    return variational_posterior(mu=mu, rho=rho)


@disk_memoize()
def laplace_approximation():
    y, point_estimate, gtab = get_preprocessed_data(force_recompute=False)
    S0_init, evals_init, evecs_init = point_estimate
    D_init = compute_D(evals_init, evecs_init).squeeze()

    prior = frozen_prior(
        alpha_S=2.0, theta_S=500.0,
        alpha_L=4.0, theta_L=2.5e-4
    )
    like = frozen_likelihood(y=y, gtab=gtab, sigma=29.0)

    L0 = np.linalg.cholesky(D_init)
    theta0 = np.array([
        np.log(S0_init if np.ndim(S0_init) == 0 else float(np.squeeze(S0_init))),
        np.log(L0[0, 0]),
        L0[1, 0],
        np.log(L0[1, 1]),
        L0[2, 0],
        L0[2, 1],
        np.log(L0[2, 2]),
    ], dtype=float)

    def unpack_theta(theta):

        S0 = np.exp(theta[0])
        L = np.array([[np.exp(theta[1]), 0.0, 0.0],
                      [theta[2], np.exp(theta[3]), 0.0],
                      [theta[4], theta[5], np.exp(theta[6])]], dtype=float)
        D = L @ L.T
        return S0, D

    def neg_log_post(theta):
        S0, D = unpack_theta(theta)

        w, V = np.linalg.eigh(D)
        order = np.argsort(w)[::-1]
        evals = w[order][None, :]
        evecs = V[:, order][None, :, :]
        S0b = np.array([S0], dtype=float)

        lp = prior.logpdf(S0b, evals=evals, evecs=evecs).squeeze()

        ll = like.logpdf(S0b, evecs, evals).squeeze()

        return -(lp + ll)

    res = minimize(neg_log_post, theta0, method='L-BFGS-B', options=dict(maxiter=500))
    theta_hat = res.x

    def numerical_hessian(f, x, h=1e-4):
        x = np.asarray(x, dtype=float)
        n = x.size
        H = np.zeros((n, n), dtype=float)
        I = np.eye(n)
        for i in range(n):
            for j in range(i, n):
                fpp = f(x + h * I[i] + h * I[j])
                fpm = f(x + h * I[i] - h * I[j])
                fmp = f(x - h * I[i] + h * I[j])
                fmm = f(x - h * I[i] - h * I[j])
                Hij = (fpp - fpm - fmp + fmm) / (4.0 * h * h)
                H[i, j] = Hij
                H[j, i] = Hij
        return H

    H = numerical_hessian(neg_log_post, theta_hat, h=1e-4)

    H = 0.5 * (H + H.T)
    jitter = 1e-8
    try:
        Sigma_theta = np.linalg.inv(H + jitter * np.eye(H.shape[0]))
    except np.linalg.LinAlgError:
        Sigma_theta = np.linalg.pinv(H + 1e-6 * np.eye(H.shape[0]))

    return mvn_reparameterized(theta_mean=theta_hat, theta_cov=Sigma_theta)


def summarize_laplace(S0_samples, evals_samples, evecs_samples):
    # Means
    S0_mean = np.mean(S0_samples)
    evals_mean = np.mean(evals_samples, axis=0)

    # 95% credible intervals
    S0_ci = np.percentile(S0_samples, [2.5, 97.5])
    evals_ci = np.percentile(evals_samples, [2.5, 97.5], axis=0)

    # Derived metrics
    MD = dti.mean_diffusivity(evals_samples)
    FA = dti.fractional_anisotropy(evals_samples)

    MD_mean, MD_ci = np.mean(MD), np.percentile(MD, [2.5, 97.5])
    FA_mean, FA_ci = np.mean(FA), np.percentile(FA, [2.5, 97.5])

    return {
        "S0": (S0_mean, *S0_ci),
        "lambda1": (evals_mean[0], *evals_ci[:, 0]),
        "lambda2": (evals_mean[1], *evals_ci[:, 1]),
        "lambda3": (evals_mean[2], *evals_ci[:, 2]),
        "MD": (MD_mean, *MD_ci),
        "FA": (FA_mean, *FA_ci)
    }


"""
=============================================================================
Visualization & Experiment Runner
=============================================================================
Plotting function and the main() script to run experiments.
"""


def main():
    y, point_estimate, gtab = get_preprocessed_data(force_recompute=False)
    S0_init, evals_init, evecs_init = point_estimate
    D_init = compute_D(evals_init, evecs_init).squeeze()

    # Find principal eigenvector from DTI estimate (for plotting)
    evec_principal = evecs_init[:, 0]

    # Set random seed and number of posterior samples
    np.random.seed(0)
    n_samples = 100000
    """
    # Run Metropolis–Hastings and plot results

    S0_mh, evals_mh, evecs_mh = metropolis_hastings(
    n_samples*10, 
    y, 
    gtab, 
    S0_init, 
    D_init, 
    gamma_prop_param=0.035,  
    nu=476,       
    plot_traces=False
    )
    burn_in = 2000
    plot_results(S0_mh[burn_in:], evals_mh[burn_in:], evecs_mh[burn_in:,:,:], evec_principal, method="mh")

    plt.show()
    print("Done.")

    # --- Variational Inference only ---
    posterior_vi = variational_inference(force_recompute=True, max_iters=1200, K=24, learning_rate=0.01)
    S0_vi, evals_vi, evecs_vi = posterior_vi.rvs(size=n_samples)

    # Plot & save
    plot_results(S0_vi, evals_vi, evecs_vi, evec_principal, method="vi")

    # ---- summaries ----


    def ci(a, level=0.95):
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        q = np.quantile(a, [lo, hi])
        return float(q[0]), float(q[1])

    S0_mean = float(np.mean(S0_vi)); S0_lo, S0_hi = ci(S0_vi)
    md = mean_diffusivity(evals_vi).squeeze()
    fa = fractional_anisotropy(evals_vi).squeeze()
    md_mean, (md_lo, md_hi) = float(np.mean(md)), ci(md)
    fa_mean, (fa_lo, fa_hi) = float(np.mean(fa)), ci(fa)

    vref = evec_principal
    projs = (evecs_vi[:, :, 2] * vref[None, :]).sum(axis=1)
    projs = np.clip(np.abs(projs), -1.0, 1.0)
    phi = 360.0 / (2 * np.pi) * np.arccos(projs)
    phi_mean, (phi_lo, phi_hi) = float(np.mean(phi)), ci(phi)

    lam1, lam2, lam3 = evals_vi[:, 2], evals_vi[:, 1], evals_vi[:, 0]
    l1_mean, (l1_lo, l1_hi) = float(np.mean(lam1)), ci(lam1)
    l2_mean, (l2_lo, l2_hi) = float(np.mean(lam2)), ci(lam2)
    l3_mean, (l3_lo, l3_hi) = float(np.mean(lam3)), ci(lam3)

    print("\n=== VI summary")
    print(f"S0: mean={S0_mean:.3f}, 95% CI=[{S0_lo:.3f}, {S0_hi:.3f}]")
    print(f"λ1 (largest): mean={l1_mean:.6f}, 95% CI=[{l1_lo:.6f}, {l1_hi:.6f}]")
    print(f"λ2:           mean={l2_mean:.6f}, 95% CI=[{l2_lo:.6f}, {l2_hi:.6f}]")
    print(f"λ3 (smallest):mean={l3_mean:.6f}, 95% CI=[{l3_lo:.6f}, {l3_hi:.6f}]")
    print(f"MD: mean={md_mean:.6f}, 95% CI=[{md_lo:.6f}, {md_hi:.6f}]")
    print(f"FA: mean={fa_mean:.3f}, 95% CI=[{fa_lo:.3f}, {fa_hi:.3f}]")
    print(f"φ (deg): mean={phi_mean:.2f}, 95% CI=[{phi_lo:.2f}, {phi_hi:.2f}]")

    print("Done.")

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------


    # Run Laplace Approximation and plot results
    posterior_laplace = laplace_approximation(force_recompute=False)
    S0_laplace, evals_laplace, evecs_laplace = posterior_laplace.rvs(size=n_samples)

    # >>> Add this summary step <<<
    stats = summarize_laplace(S0_laplace, evals_laplace, evecs_laplace)
    print("Laplace summary statistics:")
    for k, v in stats.items():
        print(f"{k}: mean={v[0]:.6f}, 95% CI=({v[1]:.6f}, {v[2]:.6f})")

    # Keep your plotting
    plot_results(S0_laplace, evals_laplace, evecs_laplace, evec_principal, method="laplace")

    print("Done.")
    """

    # Run Importance Sampling and plot results
    """
    gam, nu, highess, skitdata = IS_hyperparam_bayesian_optimization(y=y,gtab=gtab,S0_init=S0_init,D_init=D_init).optimize()
    print(gam)
    print(nu)
    """
    # Bayesian optimization resulted in gamma = 0.11 and nu = 4 for standard IS

    w_is, S0_is, evals_is, evecs_is = importance_sampling(
        y=y,
        gtab=gtab,
        S0_init=S0_init,
        D_init=D_init,
        n_samples=n_samples,
        IS_gamma_param=0.5,
        IS_nu_param=3,
        enable_logweights=False,
        enable_sequential_mc=True,
        n_sequential_mc_iterations=10,
        n_kernel_iterations=2,
        smc_evals_std=0.0001,
        smc_rot_std=2,
        smc_gamma_param=0.02
    )

    for i in range(n_samples):
        idx = np.argsort(evals_is[i])
        evals_is[i] = evals_is[i, idx]
        evecs_is[i, :, :] = evecs_is[i, :, idx]

    print(f'\n\n\nESS = {1 / np.sum(np.square(w_is))}')
    plot_results(S0_is, evals_is, evecs_is, evec_principal, weights=w_is, method="is")

    def ci(a, level=0.95):
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        q = np.quantile(a, [lo, hi])
        return float(q[0]), float(q[1])

    S0_mean = float(np.mean(S0_is));
    S0_lo, S0_hi = ci(S0_is)
    md = mean_diffusivity(evals_is).squeeze()
    fa = fractional_anisotropy(evals_is).squeeze()
    md_mean, (md_lo, md_hi) = float(np.mean(md)), ci(md)
    fa_mean, (fa_lo, fa_hi) = float(np.mean(fa)), ci(fa)

    vref = evec_principal
    projs = (evecs_is[:, :, 2] * vref[None, :]).sum(axis=1)
    projs = np.clip(np.abs(projs), -1.0, 1.0)
    phi = 360.0 / (2 * np.pi) * np.arccos(projs)
    phi_mean, (phi_lo, phi_hi) = float(np.mean(phi)), ci(phi)

    lam1, lam2, lam3 = evals_is[:, 2], evals_is[:, 1], evals_is[:, 0]
    l1_mean, (l1_lo, l1_hi) = float(np.mean(lam1)), ci(lam1)
    l2_mean, (l2_lo, l2_hi) = float(np.mean(lam2)), ci(lam2)
    l3_mean, (l3_lo, l3_hi) = float(np.mean(lam3)), ci(lam3)

    print("\n=== SMC summary")
    print(f"S0: mean={S0_mean:.3f}, 95% CI=[{S0_lo:.3f}, {S0_hi:.3f}]")
    print(f"λ1 (largest): mean={l1_mean:.6f}, 95% CI=[{l1_lo:.6f}, {l1_hi:.6f}]")
    print(f"λ2:           mean={l2_mean:.6f}, 95% CI=[{l2_lo:.6f}, {l2_hi:.6f}]")
    print(f"λ3 (smallest):mean={l3_mean:.6f}, 95% CI=[{l3_lo:.6f}, {l3_hi:.6f}]")
    print(f"MD: mean={md_mean:.6f}, 95% CI=[{md_lo:.6f}, {md_hi:.6f}]")
    print(f"FA: mean={fa_mean:.3f}, 95% CI=[{fa_lo:.3f}, {fa_hi:.3f}]")
    print(f"φ (deg): mean={phi_mean:.2f}, 95% CI=[{phi_lo:.2f}, {phi_hi:.2f}]")

    print("Done.")


def plot_results(S0, evals, evecs, evec_ref, weights=None, method=""):
    if weights is None:
        weights = np.ones_like(S0)
        weights /= np.sum(weights)

    # Choose number of bins based on sample size
    n_bins = 125

    # Squeeze arrays for plotting
    weights = weights.squeeze()
    S0 = S0.squeeze()
    md = dti.mean_diffusivity(evals).squeeze()
    fa = dti.fractional_anisotropy(evals).squeeze()

    # Compute acute angle between estimated and reference eigenvectors
    angle = 360 / (2 * np.pi) * np.arccos(np.abs(np.dot(evecs[:, :, 2], evec_ref)))

    # Create 2x2 grid of histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=False)

    axes[0, 0].hist(S0, bins=n_bins, density=True, weights=weights,
                    alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel("S0")
    axes[0, 0].set_ylabel("Density")

    axes[0, 1].hist(md, bins=n_bins, density=True, weights=weights,
                    alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel("Mean diffusivity")
    axes[0, 1].set_ylabel("Density")

    axes[1, 0].hist(fa, bins=n_bins, density=True, weights=weights,
                    alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel("Fractional anisotropy")
    axes[1, 0].set_ylabel("Density")

    axes[1, 1].hist(angle, bins=n_bins, density=True, weights=weights,
                    alpha=0.7, color='magenta', edgecolor='black')
    axes[1, 1].set_xlabel("Acute angle")
    axes[1, 1].set_ylabel("Density")

    # Adjust layout and save figure with method name
    plt.tight_layout()
    plt.savefig("results_{}.png".format(method), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()