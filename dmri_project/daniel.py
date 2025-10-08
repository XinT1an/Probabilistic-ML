
import pandas as pd
class IS_hyperparam_bayesian_optimization:
    def __init__(self, y, gtab, S0_init, D_init):
        self.y = y
        self.gtab = gtab
        self.S0_init = S0_init
        self.D_init = D_init

    def objective_function_ess(self, logweights):
        return np.exp(-logsumexp(2*logweights))

    def acquisition_function(self, best_ess_value, mean, std):
        # Expected improvement acquisition function

        #preventing division by 0
        if std < 10**-6:
            std = 10**-6

        cdf = norm.cdf(x = (mean - best_ess_value)/std)
        pdf = norm.pdf(x = (mean - best_ess_value)/std)

        return (mean - best_ess_value)*cdf + std*pdf

    def kernel(self, reference_data, new_data, std_kernel = 2):
        # Function calculates a Gaussian kernel
        gamma_vec_ref = reference_data["gamma"].to_numpy()
        nu_vec_ref = reference_data["nu"].to_numpy()

        gamma_vec_new = new_data["gamma"].to_numpy()
        nu_vec_new = new_data["nu"].to_numpy()

        num_ref_data = len(gamma_vec_ref)
        num_new_data = len(gamma_vec_new)

        # Initialize kernel matrix
        kernel_mat = np.zeros([num_ref_data, num_new_data])
        denominator = 2*std_kernel**2
        for i in range(num_ref_data):
            ref_point = np.array([gamma_vec_ref[i], nu_vec_ref[i]])
            for k in range(num_new_data):
                new_point = np.array([gamma_vec_new[k], nu_vec_new[k]])
                dist = np.linalg.norm(ref_point-new_point)
                kernel_mat[i, k] = np.exp(-np.square(dist)/denominator)

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
        gamma_initial_vec = np.random.uniform(0, 10, size = num_initial_samples)
        nu_initial_vec = np.random.uniform(nu_min, nu_max, size = num_initial_samples).astype(int)
        ess_gamma_nu_list = []
        for i, nu_sample in enumerate(nu_initial_vec):
            gamma_sample = gamma_initial_vec[i]
            logweights, _, _, _ = importance_sampling(y=self.y, gtab=self.gtab, S0_init=self.S0_init,
                                                      D_init=self.D_init, n_samples=10000, IS_gamma_param=gamma_sample,
                                                      IS_nu_param=nu_sample, enable_logweights=True)
            ess = self.objective_function_ess(logweights = logweights)
            ess_gamma_nu_list.append({"ess":ess, "gamma":gamma_sample, "nu":nu_sample})

        evaluated_data = pd.DataFrame(ess_gamma_nu_list)

        nu_vec = np.arange(nu_min, nu_max + 1)
        for i in range(num_iterations):
            gamma_sample_vec = np.random.exponential(scale = 1.5, size = num_gamma_samples)

            gen_data_list = []
            for nu_sample in nu_vec:
                for gamma_sample in gamma_sample_vec:
                    gen_data_list.append({"gamma":gamma_sample, "nu":nu_sample})

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

            logweights, _, _, _ = importance_sampling(y=self.y, gtab=self.gtab, S0_init=self.S0_init,
                                                      D_init=self.D_init, n_samples=10000, IS_gamma_param=new_gamma,
                                                      IS_nu_param=new_nu, enable_logweights=True)
            ess = self.objective_function_ess(logweights = logweights)
            evaluated_data = pd.concat([evaluated_data,pd.DataFrame({"ess":[ess], "gamma":[new_gamma], "nu":[new_nu]})],
                                       ignore_index = True)

        highest_ess_index = evaluated_data["ess"].idxmax()
        highest_ess_value = evaluated_data.loc[highest_ess_index, "ess"]
        best_gamma = evaluated_data.loc[highest_ess_index, "gamma"]
        best_nu = evaluated_data.loc[highest_ess_index, "nu"]

        return best_gamma, best_nu, highest_ess_value, evaluated_data


    # ----------------------------------------------------------------------------

def importance_sampling(y,
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
                        smc_evals_std=0.00005,
                        smc_rot_std=1,
                        smc_gamma_param=0.02
                        ):

    def IS_algorithm():
        # Generating samples
        S0_samples = gamma.rvs(a=IS_gamma_param ** -2, scale=(IS_gamma_param ** 2) * S0_init, size=n_samples)
        log_qS0 = gamma.logpdf(x=S0_samples, a=IS_gamma_param ** -2, scale=(IS_gamma_param ** 2) * S0_init)
        D_samples = wishart.rvs(df=IS_nu_param, scale=D_init, size=n_samples)

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

        def residual_resampling(logweights, S0_samples, evals_samples, evecs_samples):
            # Resampling of particles based on residual resampling
            S0_resampled = np.zeros(np.shape(S0_samples))
            evals_resampled = np.zeros(np.shape(evals_samples))
            evecs_resampled = np.zeros(np.shape(evecs_samples))

            # Deterministic part (sample all with weights > 1/n_samples, n_i[sample] times)
            n_i = np.floor(n_samples * np.exp(logweights)).astype(int)
            indices = np.repeat(a=np.arange(n_samples), repeats=n_i)

            S0_resampled[:np.sum(n_i)] = S0_samples[indices]
            evals_resampled[:np.sum(n_i), :] = evals_samples[indices, :]
            evecs_resampled[:np.sum(n_i), :, :] = evecs_samples[indices, :, :]

            # Resample the residuals stochastichally (multinomial)
            m = n_samples - np.sum(n_i)
            residual_weights = np.exp(logweights) - n_i / n_samples
            residual_weights = residual_weights / np.sum(residual_weights)
            indices = np.random.choice(a=n_samples, size=m, p=residual_weights)

            S0_resampled[np.sum(n_i):] = S0_samples[indices]
            evals_resampled[np.sum(n_i):, :] = evals_samples[indices, :]
            evecs_resampled[np.sum(n_i):, :, :] = evecs_samples[indices, :, :]

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
            rotation_angles_sample = norm.rvs(loc=0, scale=std, size=3)
            rotation_matrix = Rotation.from_euler('XYZ', rotation_angles_sample, degrees=True).as_matrix()
            return rotation_matrix @ evecs_mean

        def evec_rotation_logpdf(x, loc, std):
            # Returns the logpdf of rotation around yaw, pitch and roll given an std in degrees assuming gaussian.
            x_loc_rotation_matrix = x @ loc.T
            angles_difference = Rotation.from_matrix(x_loc_rotation_matrix).as_euler('XYZ', degrees=True)
            angles_difference = (angles_difference + 180) % 360 - 180
            return np.sum(norm.logpdf(x=angles_difference, loc=0, scale=std))

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

                    # Calculating the acceptance prob for the proposed samples
                    log_pi_z_prime = (prior.logpdf(S0_prime, evecs_prime, evals_prime) +
                                      phi * likelihood.logpdf(S0_prime, evecs_prime, evals_prime)
                                      )

                    log_q_z = (gamma.logpdf(x=S0, a=smc_gamma_param ** -2, scale=(smc_gamma_param ** 2) * S0_prime) +
                               np.sum(norm.logpdf(x=evals, loc=evals_prime, scale=smc_evals_std)) +
                               evec_rotation_logpdf(x=evecs, loc=evecs_prime, std=smc_rot_std)
                               )

                    log_pi_z = prior.logpdf(S0, evecs, evals) + phi * likelihood.logpdf(S0, evecs, evals)

                    log_q_z_prime = (gamma.logpdf(x=S0_prime, a=smc_gamma_param ** -2, scale=(smc_gamma_param ** 2) * S0) +
                                     np.sum(norm.logpdf(x=evals_prime, loc=evals, scale=smc_evals_std)) +
                                     evec_rotation_logpdf(x=evecs_prime, loc=evecs, std=smc_rot_std)
                                     )

                    log_acceptance_prob = (log_pi_z_prime + log_q_z - log_pi_z - log_q_z_prime)[0]

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

        # RUNNING THE ALGORITHM
        # Initializing the annealment vector
        phi_vec = np.zeros(n_sequential_mc_iterations)

        # Initializing by sampling from the prior
        prior = frozen_prior()
        likelihood = frozen_likelihood(y=y, gtab=gtab)

        S0_samples, evals_samples, evecs_samples = prior.rvs(size=n_samples)
        logweights = np.log(np.ones(n_samples) / n_samples)  # initializing the weights (uniformely)

        for i in range(1, n_sequential_mc_iterations):
            # Calculating the new annealment exponent
            phi_vec[i] = (i / n_sequential_mc_iterations) ** 3
            delta_phi = phi_vec[i] - phi_vec[i - 1]

            loglikelihood = likelihood.logpdf(S0_samples, evecs_samples, evals_samples)

            # Updating weights and normalizing
            logweights = logweights + delta_phi * loglikelihood
            logweights = logweights - logsumexp(logweights)

            # Calculate the ess value and resample (and reset weights)
            ess = effective_sample_size(logweights)
            if ess < n_samples / 2:
                S0_samples, evals_samples, evecs_samples = residual_resampling(logweights,
                                                                               S0_samples,
                                                                               evals_samples,
                                                                               evecs_samples
                                                                               )
                logweights = np.log(np.ones(n_samples) / n_samples)

            if acceptance_ratio > 0.5:
                gamma_param = 0.5
                rotation_std = 20
                eval_std

            # Rejuvinate with markov kernel
            S0_samples, evals_samples, evecs_samples, acceptance_ratio = markov_kernel_rejuvination(S0_samples,
                                                                                                    evals_samples,
                                                                                                    evecs_samples,
                                                                                                    phi=phi_vec[i])
            print(f'\ni = {i}')
            print(f'Acceptance ratio = {100 * acceptance_ratio:.1f}%')
            print(f'Unique S0 samples: {100 * (len(np.unique(np.round(S0_samples, 6))) / len(S0_samples)):.1f}%')

        # No need for resampling on last iteration
        delta_phi = 1 - phi_vec[-1]

        loglikelihood = likelihood.logpdf(S0_samples, evecs_samples, evals_samples)

        logweights = logweights + delta_phi * loglikelihood
        logweights = logweights - logsumexp(logweights)

        # Returning the particles and weights
        if enable_logweights == True:
            importance_weights = logweights

        elif enable_logweights == False:
            importance_weights = np.exp(logweights)

        return importance_weights, S0_samples, evals_samples, evecs_samples

    if enable_sequential_mc == False:
        return IS_algorithm()

    elif enable_sequential_mc == True:
        return SMC_algorithm()




# --------------------------------------------------------------------------------
    """
    # returned gamma as 0.11 and nu as 4
    gam, nu, _, _ = IS_hyperparam_bayesian_optimization(y=y,gtab=gtab,S0_init=S0_init,D_init=D_init).optimize()
    print(f"gamma value: {gam}")
    print(f'nu value: {nu}')
    """

    w_is, S0_is, evals_is, evecs_is = importance_sampling(y = y,
                                                          gtab = gtab,
                                                          S0_init = S0_init,
                                                          D_init = D_init,
                                                          n_samples = n_samples,
                                                          IS_gamma_param = 0.11,
                                                          IS_nu_param = 4,
                                                          enable_logweights = False,
                                                          enable_sequential_mc = True,
                                                          n_sequential_mc_iterations = 100,
                                                          n_kernel_iterations = 10,
                                                          smc_evals_std = 0.00005,
                                                          smc_rot_std = 1,
                                                          smc_gamma_param = 0.02)
                                                          )