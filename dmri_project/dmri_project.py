"""
===============================================================================
Bayesian Inference in Diffusion Tensor Imaging - Project Template
===============================================================================

This Python file provides the starter template for the course project in
"Advanced Probabilistic Machine Learning",
Department of Information Technology, Uppsala University.

Authors:
- Jens Sjölund (original author) - jens.sjolund@it.uu.se
- Anton O'Nils (updates & finalization) - anton.o-nils@it.uu.se
- Stina Brunzell (updates & finalization) - stina.brunzell@it.uu.se

------------------------------------------------------------------------------
Purpose
------------------------------------------------------------------------------
The project concerns Bayesian inference in diffusion MRI (dMRI), specifically
the diffusion tensor model (DTI). The goal is to estimate local tissue
properties (baseline signal S0 and diffusion tensor D) from real-world dMRI
measurements, using different Bayesian inference techniques.

Each student/group member will implement one of the following inference methods:
  1. Metropolis-Hastings  
  2. Importance Sampling  
  3. Variational Inference  
  4. Laplace Approximation  

The provided code gives:
  - Utilities for loading and preprocessing the "Stanford HARDI dataset". 
  - Helper functions for matrix operations, parameterizations and gradients.  
  - A skeleton structure for the prior, likelihood, and posterior approx.   
  - Placeholders where each inference method should be implemented.  
  - Plotting routines to visualize posterior summaries.

------------------------------------------------------------------------------
Dataset
------------------------------------------------------------------------------
The code uses the Stanford HARDI diffusion MRI dataset (Rokem et al., 2015),
accessible via DIPY's "get_fnames('stanford_hardi')".

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- Several classes and methods are left as "NotImplementedError"; students are
  expected to fill these in.  
- Computations are memoized with "disk_memoize" to avoid repeated costly runs.  
- Results for each inference method are automatically plotted and saved.  

=============================================================================
Imports
=============================================================================
Required libraries: numpy, matplotlib, scipy, dipy
Install with: pip install numpy matplotlib scipy dipy
"""

# Standard library: general utilities
import os
import pickle
import hashlib
from functools import wraps

# NumPy and Matplotlib: math and plotting
import numpy as np
import matplotlib.pyplot as plt

# SciPy: probability distributions, math functions, and optimization
# Hint: these tools might be useful later in the project
from scipy.stats import gamma, norm, wishart, multivariate_normal
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp, digamma
from scipy.optimize import minimize

# DIPY: diffusion MRI utilities and models
from dipy.io.image import load_nifti, save_nifti   # for loading / saving imaging datasets
from dipy.io.gradients import read_bvals_bvecs     # for loading / saving our bvals and bvecs
from dipy.core.gradients import gradient_table     # for constructing gradient table from bvals/bvecs
from dipy.data import get_fnames                   # for small datasets that we use in tests and examples
from dipy.segment.mask import median_otsu          # for masking out the background
import dipy.reconst.dti as dti                     # for diffusion tensor model fitting and metrics



"""
=============================================================================
Caching Utility (already implemented)
=============================================================================
Provides disk-based memoization to avoid recomputation.
"""

def disk_memoize(cache_dir="cache"):
    """
    Decorator for caching function outputs on disk.

    This utility is already implemented and should not be modified by students.
    It allows expensive computations to be stored and re-used across runs,
    based on the function arguments. If you call the same function again with
    the same inputs, it returns the cached results instead of recomputing.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Optionally force a fresh computation (ignores cache if True)
            force = kwargs.pop("force_recompute", False)

            # Make sure the cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Build a unique hash key from the function name and arguments
            func_name = func.__name__
            key = (func_name, args, kwargs)
            hash_str = hashlib.md5(pickle.dumps(key)).hexdigest()
            cache_path = os.path.join(cache_dir, f"{func_name}_{hash_str}.pkl")

            # Load the cached result if it exists (and recomputation is not forced)
            if not force and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            # Otherwise: compute the result, then cache it to disk
            result = func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

            return result
        
        return wrapper
    return decorator



"""
=============================================================================
Data Loading & Preprocessing (already implemented)
=============================================================================
Loads the Stanford HARDI dataset, applies masking/cropping, and
extracts one voxel with a DTI point estimate for testing.
"""

@disk_memoize()
def get_preprocessed_data():
    """
    Load and preprocess a single voxel of diffusion MRI data.

    What it does:
    - Loads the dataset and gradient information (b-values and b-vectors).
    - Fits a diffusion tensor model (DTI) to one voxel.
    - Extracts a point estimate: baseline signal (S0), eigenvalues, eigenvectors.

    Returns
    -------
    y : ndarray
        Observed diffusion MRI signal vector for a single voxel.
    point_estimate : [S0, evals, evecs]
        Estimated baseline signal, eigenvalues, and eigenvectors.
    gtab : GradientTable
        Gradient table with b-values (diffusion weighting strength)
        and b-vectors (gradient directions).
    """

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
    """
    Load and preprocess the Stanford HARDI diffusion MRI dataset.

    What it does:
    - Downloads the dataset if not already present (via DIPY).
    - Loads the 4D diffusion MRI volume (x, y, z, measurements).
    - Reads b-values (diffusion weighting strength) and b-vectors (gradient directions).
    - Creates a gradient table (gtab) combining this information.
    - Applies a brain mask and cropping to remove background and reduce size.

    Returns
    -------
    maskdata : ndarray
        The masked and cropped diffusion MRI data.
    mask : ndarray (boolean)
        The brain mask used to exclude background voxels.
    gtab : GradientTable
        Gradient information (b-values and b-vectors) for each measurement.
    """

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


"""
=============================================================================
Linear Algebra Helpers (already implemented)
=============================================================================
Functions for reconstructing tensors and switching between
parameterizations. Already implemented.
Hint: you will make use of these helpers later in the project,
the ones involving theta are useful for VI and Laplace.
"""

def compute_D(evals, V):
    """
    Reconstruct the diffusion tensor D from eigenvalues and eigenvectors.

    D = V Λ V.T, where Λ is the diagonal matrix of eigenvalues.

    Parameters
    ----------
    evals : ndarray
        Eigenvalues, shape (3,) or batched.
    V : ndarray
        Eigenvectors, shape (3, 3) or batched.

    Returns
    -------
    D : ndarray
        Diffusion tensor(s), shape (..., 3, 3).
    """

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
    """
    Convert a diffusion tensor D into an unconstrained parameter vector theta.

    Follows Eq. (18): D = L L.T with L from Cholesky factorization.
    Diagonals are log-transformed, off-diagonals kept raw.

    Parameters
    ----------
    D : ndarray (3, 3)
        Symmetric positive-definite diffusion tensor.

    Returns
    -------
    theta : ndarray (6,)
        Unconstrained parameter vector corresponding to the lower-triangular
        entries of L (log of diagonals, raw off-diagonals).
    """
    
    # Compute Cholesky factor (lower-triangular L) of D
    L = np.linalg.cholesky(D)
    
    # Indices of lower-triangular entries (including diagonal)
    p = D.shape[0]
    tril_indices = np.tril_indices(p)
    theta = []

    # Store log of diagonal entries, raw off-diagonal entries
    for i, j in zip(*tril_indices):
        if i == j:
            theta.append(np.log(L[i, j]))   # Diagonal: log-transform
        else:
            theta.append(L[i, j])           # Off-diagonal: raw value

    return np.array(theta)


def D_from_theta(theta):
    """
    Convert unconstrained parameter vector theta back into diffusion tensor D.

    Follows Eq. (18): D = L L.T with L constructed from theta.
    Diagonal entries of L are exponentiated to ensure positivity,
    off-diagonals are used as raw values.

    Parameters
    ----------
    theta : ndarray (..., 6)
        Unconstrained parameters corresponding to the lower-triangular
        entries of L (log-diagonals, raw off-diagonals).

    Returns
    -------
    D : ndarray (..., 3, 3)
        Symmetric positive-definite diffusion tensor(s).
    """
    
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
            L[..., i, j] = np.exp(theta[..., k])   # Diagonal
        else:
            L[..., i, j] = theta[..., k]           # Off-diagonal

    # Reconstruct D = L @ L.T (batch-aware matrix multiplication)
    D = L @ np.swapaxes(L, -1, -2)

    return D.squeeze()


def grad_D_wrt_theta_at_D(D):
    """
    Compute nabla_theta D evaluated at D.

    Uses the parameterization in Eq. (18): D = L L.T where L is built from 
    theta. Returns the gradient tensor with one (3x3) slice per theta component.

    Parameters
    ----------
    D : ndarray (3, 3)
        Symmetric positive-definite diffusion tensor.

    Returns
    -------
    grad_D : ndarray (3, 3, 6)
        Gradient of D w.r.t. theta, one 3x3 matrix per parameter.
    """
    
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


"""
=============================================================================
Bayesian Model Components (need to be implemented)
=============================================================================
Students: implement all parts in this section (priors, likelihoods, etc.)
These are required before any inference method can be attempted.
"""

class frozen_prior:
    def __init__(self,
                    alpha_S=2.0, theta_S=500.0,    
                    alpha_L=4.0, theta_L=2.5e-4,   
                    random_state=None):
            # Store hyperparameters
            self.alpha_S = float(alpha_S)
            self.theta_S = float(theta_S)
            self.alpha_L = float(alpha_L)
            self.theta_L = float(theta_L)

            self._gamma_S0 = gamma(a=self.alpha_S, scale=self.theta_S)
            self._gamma_L = gamma(a=self.alpha_L, scale=self.theta_L)

            self.random_state = (np.random.RandomState(random_state)
                                if random_state is not None else np.random)

    def rvs(self, size=1):
        # draw prior samples
        size = int(size)
        # S0 = Gamma(alpha_S, theta_S)
        S0 = self._gamma_S0.rvs(size=size, random_state=self.random_state).astype(float)
        # Lambdai = iid Gamma(alpha_Lambda, theta_Lambda)
        # eigen values
        evals = self._gamma_L.rvs(size=(size, 3), random_state=self.random_state).astype(float)
        # V= Uniform(S0(3))
        Vs = Rotation.random(size=size, random_state=self.random_state).as_matrix().astype(float)

        order = np.argsort(evals, axis=1)[:, ::-1]
        evals_sorted = np.take_along_axis(evals, order, axis=1) # eigenvalues sorted in descending order

        # columns of V match the sorted eigenvalues
        evecs_sorted = np.stack(
        [Vs[i][:, order[i]] for i in range(size)],
        axis=0
        )
        return S0.reshape(size,), evals_sorted.reshape(size, 3), evecs_sorted.reshape(size, 3, 3)
                 
    
    def logpdf(self, S0, evecs=None, evals=None, D=None):
        S0 = np.asarray(S0, dtype=float).reshape(-1)
        evals = np.asarray(evals, dtype=float).reshape(-1, 3)
        # ensure batch sizes line up
        B = max(S0.shape[0], evals.shape[0])
        if S0.shape[0] == 1 and B > 1:
            S0 = np.repeat(S0, B)
        if evals.shape[0] == 1 and B > 1:
            evals = np.repeat(evals, B, axis=0)
        # log-gamma for S0 and each lambda
        lp_S0 = self._gamma_S0.logpdf(S0)
        lp_l = np.sum(self._gamma_L.logpdf(evals), axis=1)
        return (lp_S0 + lp_l).reshape(B,)
        


class frozen_likelihood:
    # Placeholder for the likelihood (with partial code provided).
    # Hint: you may want to add input parameters to these methods.

    def __init__(self, y, gtab, sigma=29):
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.gtab = gtab   # store gradient table with b-values and b-vectors
        self.sigma2 = float(sigma) ** 2
        self.n = self.y.size
        #norm constant
        self._log_norm_const = -0.5*self.n*np.log(2.0*np.pi*self.sigma2)
      
    def logpdf(self, S0, evecs, evals):
        S0 = np.atleast_1d(S0)        # ensure S0 is array
        D = compute_D(evals, evecs)   # reconstruct diffusion tensor

        q = np.sqrt(self.gtab.bvals[:, None]) * self.gtab.bvecs

        S = S0[:, None] * np.exp( - np.einsum('...j, ijk, ...k->i...', q, D, q))

        S = np.atleast_2d(S)
        y = self.y[None, :]
        resid = y - S
        sq = np.sum(resid*resid, axis=1)
        ll = self._log_norm_const - 0.5 *sq/self.sigma2
        return ll.squeeze()
        
        raise NotImplementedError



"""
=============================================================================
Posterior Approximations (need to be implemented)
=============================================================================
Students: implement these approximations, which are only used in the
corresponding inference methods below:
  - variational_posterior: used only for Variational Inference
  - mvn_reparameterized: used only for Laplace Approximation

They are NOT needed for Metropolis-Hastings or Importance Sampling.
"""

class variational_posterior:
    # Placeholder for variational posterior approximation.
    # Hint: you may want to add input parameters to these methods.
    # The score() method is already implemented and can be used later
    # when implementing inference (with REINFORCE leave-one-out estimator).

    def __init__(self):
        raise NotImplementedError

    def logpdf(self):
        raise NotImplementedError
    
    def rvs(self, size):
        raise NotImplementedError

        return S0_samples, evals_samples, evecs_samples

    def score(self, S0, D):
        # Combine score contributions from gamma and Wishart parts
        score_wrt_log_shape, score_wrt_log_scale = self.gamma_score(S0)
        score_wrt_theta, score_wrt_log_df = self.wishart_score(D)
        return np.concatenate([
            score_wrt_log_shape, score_wrt_log_scale, score_wrt_theta, score_wrt_log_df]
        )

    def gamma_score(self, x):
        # Score function for gamma distribution
        score_wrt_log_shape = (np.log(x / self.scale) - digamma(self.shape)) * self.shape
        score_wrt_log_scale = (x / self.scale**2 - self.shape / self.scale) * self.scale
        return score_wrt_log_shape, score_wrt_log_scale

    def wishart_score(self, D):
        # Score function for Wishart distribution
        W = self.df * D
        Sigma_inv = np.linalg.inv(self.Sigma)
        score_wrt_Sigma = 0.5 * Sigma_inv @ (W - self.df * self.Sigma) @ Sigma_inv
        score_wrt_theta = np.tensordot(
            score_wrt_Sigma, grad_D_wrt_theta_at_D(self.Sigma), axes=([0,1], [0,1])
        )
        p = W.shape[0]
        _, logdet_W = np.linalg.slogdet(W)
        _, logdet_Sigma = np.linalg.slogdet(self.Sigma)
        digamma_sum = np.sum([digamma((self.df + 1 - j) / 2.0) for j in range(1, p+1)])
        score_wrt_log_df = ((self.df - 2) / 2) * (logdet_W - p * np.log(2) - logdet_Sigma - digamma_sum)
        return score_wrt_theta, score_wrt_log_df


class mvn_reparameterized:
    # Placeholder for multivariate normal approximation.
    # Hint: you may want to add input parameters to these methods.
    
    def __init__(self, theta_mean, theta_cov):
        self.theta_mean = np.asarray(theta_mean, dtype=float).reshape(-1)
        self.theta_cov = np.asarray(theta_cov, dtype=float)

    
        assert self.theta_mean.shape[0] == 7 
        assert self.theta_cov.shape == (7, 7) 
        
    
    def rvs(self, size):
        size = int(size)
        # Sample θ from N(θ̂, Σθ)
        thetas = np.random.multivariate_normal(self.theta_mean, self.theta_cov, size=size)  # (size, 7)

        # Allocate outputs
        S0_samples   = np.zeros((size,), dtype=float)
        evals_samples = np.zeros((size, 3), dtype=float)
        evecs_samples = np.zeros((size, 3, 3), dtype=float)

        for i in range(size):
            t = thetas[i]

            # Map θ -> S0 and L (lower-triangular with exp on diagonals)
            S0 = np.exp(t[0])
            L = np.array([[np.exp(t[1]), 0.0,           0.0],
                          [t[2],         np.exp(t[3]),  0.0],
                          [t[4],         t[5],          np.exp(t[6])]], dtype=float)
            D = L @ L.T

            # Eigendecompose D, sort eigenvalues descending, reorder eigenvectors accordingly
            w, V = np.linalg.eigh(D)
            order = np.argsort(w)[::-1]
            evals = w[order]
            evecs = V[:, order]

            # Store
            S0_samples[i]    = S0
            evals_samples[i] = evals
            evecs_samples[i] = evecs

    
        return S0_samples, evals_samples, evecs_samples


"""
=============================================================================
Inference Methods (need to be implemented)
=============================================================================
Students: implement one method each (MH, IS, VI, or Laplace).
Uses memoization to speed up repeated runs.
"""

@disk_memoize()
def metropolis_hastings(n_samples, y, gtab, S0_init, D_init, gamma_prop_param, theta_prop_scale, plot_traces=False):
    """
    Metropolis-Hastings sampler for DTI Bayesian model.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    y : ndarray
        Observed diffusion signal
    gtab : GradientTable
        Gradient information
    S0_init : float
        Initial value for S0
    D_init : ndarray (3, 3)
        Initial diffusion tensor
    gamma_prop_param : float
        Proposal scale for Gamma-distributed parameters (S0, eigenvalues)
    theta_prop_scale : float
        Proposal scale for theta parameters (rotation)
    plot_traces : bool
        Whether to plot trace plots for debugging

    Returns
    -------
    S0_samples : ndarray (n_samples,)
        Samples of S0 parameter
    evals_samples : ndarray (n_samples, 3)
        Samples of eigenvalues
    evecs_samples : ndarray (n_samples, 3, 3)
        Samples of eigenvectors
    """
    
    # Initialize prior and likelihood
    prior = frozen_prior()
    likelihood = frozen_likelihood(y=y, gtab=gtab, sigma=29.0)
    
    # Get initial eigenvalues and eigenvectors from D_init

    evals_init, evecs_init = np.linalg.eigh(D_init)
    
    # Sort eigenvalues and rearrange eigenvectors accordingly
    idx = evals_init.argsort()[::-1]  
    current_evals = evals_init[idx]
    current_evecs = evecs_init[:, idx]
    current_S0 = S0_init
    
    # Storage for samples
    S0_samples = np.zeros(n_samples)
    evals_samples = np.zeros((n_samples, 3))
    evecs_samples = np.zeros((n_samples, 3, 3))
    
    # Calculate current log-posterior
    current_log_prior = prior.logpdf(current_S0, evals=current_evals, evecs=current_evecs)
    current_log_likelihood = likelihood.logpdf(current_S0, current_evecs, current_evals)
    current_log_posterior = current_log_prior + current_log_likelihood
    
    # Acceptance counters
    accept_S0 = 0
    accept_evals = 0
    accept_evecs = 0
    total_S0_proposals = 0
    total_evals_proposals = 0
    total_evecs_proposals = 0
    
    print("Starting Metropolis-Hastings sampling...")
    
    for i in range(n_samples):
        if i % 10000 == 0 and i > 0:
            print(f"Sample {i}/{n_samples}, Acceptance rates: "
                  f"S0: {accept_S0/total_S0_proposals:.3f}, "
                  f"Evals: {accept_evals/total_evals_proposals:.3f}, "
                  f"Evecs: {accept_evecs/total_evecs_proposals:.3f}")

        # --- Block 1: Propose new S0 ---
        total_S0_proposals += 1
        # Log-normal proposal for S0 (ensures positivity)
        proposed_S0 = current_S0 * np.exp(gamma_prop_param * np.random.randn())
        
        proposed_log_prior = prior.logpdf(proposed_S0, evals=current_evals, evecs=current_evecs)
        # Likelihood only needs to be recomputed for S0
        proposed_log_likelihood = likelihood.logpdf(proposed_S0, current_evecs, current_evals)
        proposed_log_posterior = proposed_log_prior + proposed_log_likelihood
        
        # Log acceptance ratio for S0 (symmetric proposal in log-space)
        log_alpha_S0 = proposed_log_posterior - current_log_posterior
        
        if np.log(np.random.rand()) < log_alpha_S0:
            current_S0 = proposed_S0
            current_log_posterior = proposed_log_posterior
            accept_S0 += 1
        
        # --- Block 2: Propose new eigenvalues (evals) ---
        total_evals_proposals += 1
        # Log-normal proposal for evals (ensures positivity)
        proposed_evals_unsorted = current_evals * np.exp(gamma_prop_param * np.random.randn(3))
        
        # --- CRITICAL FIX: Sort the proposed eigenvalues (descending) ---
        # Get the indices that sort the array
        idx_evals = proposed_evals_unsorted.argsort()[::-1]
        proposed_evals = proposed_evals_unsorted[idx_evals]
        
        proposed_log_prior = prior.logpdf(current_S0, evals=proposed_evals, evecs=current_evecs)
        # Likelihood uses the new, sorted evals
        proposed_log_likelihood = likelihood.logpdf(current_S0, current_evecs, proposed_evals)
        proposed_log_posterior = proposed_log_prior + proposed_log_likelihood
        
        # Log acceptance ratio for eigenvalues
        log_alpha_evals = proposed_log_posterior - current_log_posterior
        
        if np.log(np.random.rand()) < log_alpha_evals:
            # --- CRITICAL FIX: Re-order the eigenvectors to match the new evals ---
            # The current_evecs are ordered to match current_evals.
            # To match the new, sorted proposed_evals, we must re-order the columns
            # of current_evecs using the same sorting indices.
            current_evecs = current_evecs[:, idx_evals]
            
            current_evals = proposed_evals
            current_log_posterior = proposed_log_posterior
            accept_evals += 1
        
        # --- Block 3: Propose new eigenvectors (evecs) ---
        total_evecs_proposals += 1
        
        # Generate small random rotation
        # Use a Gaussian distribution for axis-angle/Rodrigues vector for small steps
        rotvec = theta_prop_scale * np.random.randn(3)
        R = Rotation.from_rotvec(rotvec).as_matrix()
        
        # Apply rotation to current eigenvectors
        # Use R.T or R depending on convention, sticking to your original R.T
        proposed_evecs = current_evecs @ R.T
        
        proposed_log_prior = prior.logpdf(current_S0, evals=current_evals, evecs=proposed_evecs)
        # Likelihood uses the new evecs
        proposed_log_likelihood = likelihood.logpdf(current_S0, proposed_evecs, current_evals)
        proposed_log_posterior = proposed_log_prior + proposed_log_likelihood
        
        # Log acceptance ratio for eigenvectors (symmetric proposal on SO(3) for small steps)
        log_alpha_evecs = proposed_log_posterior - current_log_posterior
        
        if np.log(np.random.rand()) < log_alpha_evecs:
            current_evecs = proposed_evecs
            current_log_posterior = proposed_log_posterior
            accept_evecs += 1
        
        # Store sample
        S0_samples[i] = current_S0
        evals_samples[i] = current_evals
        evecs_samples[i] = current_evecs
    
    # Print final acceptance rates
    print(f"Final acceptance rates:")
    print(f"S0: {accept_S0/total_S0_proposals:.3f}")
    print(f"Eigenvalues: {accept_evals/total_evals_proposals:.3f}")
    print(f"Eigenvectors: {accept_evecs/total_evecs_proposals:.3f}")
    

    return S0_samples, evals_samples, evecs_samples



@disk_memoize()
def importance_sampling(n_samples, gamma_param, nu_param):
    # Students: implement Importance Sampling here.
    # Before starting, make sure the prior and likelihood are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    raise NotImplementedError

    return importance_weights, S0_samples, evals_samples, evecs_samples


@disk_memoize()
def variational_inference(max_iters, K, learning_rate):
    # Students: implement Variational Inference here.
    # Before starting, make sure the prior, likelihood and variational_posterior are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    raise NotImplementedError

    return variational_posterior(...)


@disk_memoize()
def laplace_approximation():
    # Students: implement the Laplace Approximation here.
    # Before starting, make sure the prior, likelihood and mvn_reparameterized are implemented.
    # Note: you may change, add, or remove input parameters depending on your design
    # (e.g. pass initialization values like those prepared in main()).

    # Data, prior, likelihood
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
        np.log(S0_init if np.ndim(S0_init)==0 else float(np.squeeze(S0_init))),
        np.log(L0[0, 0]),
        L0[1, 0],
        np.log(L0[1, 1]),
        L0[2, 0],
        L0[2, 1],
        np.log(L0[2, 2]),
    ], dtype=float)

    def unpack_theta(theta):
        
        S0 = np.exp(theta[0])
        L = np.array([[np.exp(theta[1]), 0.0,             0.0],
                      [theta[2],         np.exp(theta[3]), 0.0],
                      [theta[4],         theta[5],         np.exp(theta[6])]], dtype=float)
        D = L @ L.T
        return S0, D

    def neg_log_post(theta):
        S0, D = unpack_theta(theta)

        w, V = np.linalg.eigh(D)
        order = np.argsort(w)[::-1]
        evals = w[order][None, :]         
        evecs = V[:, order][None, :, :]   
        S0b   = np.array([S0], dtype=float)
 
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
                fpp = f(x + h*I[i] + h*I[j])
                fpm = f(x + h*I[i] - h*I[j])
                fmp = f(x - h*I[i] + h*I[j])
                fmm = f(x - h*I[i] - h*I[j])
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

import numpy as np
import dipy.reconst.dti as dti

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
        "lambda1": (evals_mean[0], *evals_ci[:,0]),
        "lambda2": (evals_mean[1], *evals_ci[:,1]),
        "lambda3": (evals_mean[2], *evals_ci[:,2]),
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
    # Initialize with preprocessed data and DTI point estimate
    # (these values can be used as starting points for inference methods)
    y, point_estimate, gtab = get_preprocessed_data(force_recompute=False)
    S0_init, evals_init, evecs_init = point_estimate
    D_init = compute_D(evals_init, evecs_init).squeeze()

    # Find principal eigenvector from DTI estimate (for plotting)
    evec_principal = evecs_init[:, 0]

    # Set random seed and number of posterior samples
    np.random.seed(0)
    n_samples = 10000

    # Run Metropolis–Hastings and plot results
    """ S0_mh, evals_mh, evecs_mh = metropolis_hastings(
    n_samples, 
    y, 
    gtab, 
    S0_init, 
    D_init, 
    gamma_prop_param=0.065, 
    theta_prop_scale=0.03, 
    plot_traces=False
    )

    burn_in = 5000

    plot_results(S0_mh[burn_in:], evals_mh[burn_in:], evecs_mh[burn_in:,:,:], evec_principal, method="mh") """

    # Run Importance Sampling and plot results
    """ w_is, S0_is, evals_is, evecs_is = importance_sampling(force_recompute=False)
    plot_results(S0_is, evals_is, evecs_is, evec_principal, weights=w_is, method="is")

    # Run Variational Inference and plot results
    posterior_vi = variational_inference(force_recompute=False)
    S0_vi, evals_vi, evecs_vi = posterior_vi.rvs(size=n_samples)
    plot_results(S0_vi, evals_vi, evecs_vi, evec_principal, method="vi") """

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



def plot_results(S0, evals, evecs, evec_ref, weights=None, method=""):
    """
    Plot posterior results as histograms and save to file.

    Creates histograms of baseline signal (S0), mean diffusivity (MD),
    fractional anisotropy (FA), and the angle between estimated and
    reference eigenvectors.

    Parameters
    ----------
    S0 : ndarray
        Sampled baseline signals.
    evals : ndarray
        Sampled eigenvalues of the diffusion tensor.
    evecs : ndarray
        Sampled eigenvectors of the diffusion tensor.
    evec_ref : ndarray
        Reference principal eigenvector (from point estimate).
    weights : ndarray, optional
        Importance weights for samples. Uniform if None.
    method : str
        Name of inference method (used in output filename).
    """
    
    # Use uniform weights if none provided
    if weights is None:
        weights = np.ones_like(S0)
        weights /= np.sum(weights)

    # Choose number of bins based on sample size
    n_bins = np.floor(np.sqrt(len(weights))).astype(int)

    # Squeeze arrays for plotting
    weights = weights.squeeze()
    S0 = S0.squeeze()
    md = dti.mean_diffusivity(evals).squeeze()
    fa = dti.fractional_anisotropy(evals).squeeze()

    # Compare principal (largest-eigenvalue) eigenvector to the reference
    cosine = np.einsum('bi,i->b', evecs[:, :, 0], evec_ref)   
    cosine = np.clip(np.abs(cosine), -1.0, 1.0)               
    angle = np.degrees(np.arccos(cosine))

    
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




