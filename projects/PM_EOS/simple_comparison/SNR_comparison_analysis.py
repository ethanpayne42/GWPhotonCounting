import GWPhotonCounting
import jax.numpy as jnp

frequencies = jnp.sort(jnp.fft.fftfreq(2**13, d=1/1e4))

detector_nosqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_shot_psd_nosqz.csv', 
    '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_classical_psd.csv', 
    gamma=100, random_seed=1632, N_frequency_spaces=10)

detector_sqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_total_psd_sqz.csv', None, 
    gamma=100, random_seed=1632, N_frequency_spaces=10)


LorentzianModel = GWPhotonCounting.signal.PostMergerLorentzian()
KNNModel = GWPhotonCounting.signal.PostMergerKNN(knn_file_path='/home/ethan.payne/code_libraries/apr4_knn_gw_model_2024/KNN_Models/APR4-knn_model-N100')

# Generating the strain signal
# mtot = 2.4
# phi0 = 0
# z = 0.03
# ra = jnp.pi/4
# dec = jnp.pi/4
# iota=jnp.pi/4
# psi = 0

# PM_strain = KNNModel.generate_strain(detector_nosqz, frequencies, mtot, phi0, z, ra, dec, iota, psi)
PM_strain = LorentzianModel.generate_strain(detector_nosqz, frequencies, 2.75e3, 50, 1e-22, jnp.pi/2, 0)[0]

expected_signal_photon_count = detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies)

poisson_likelihood = GWPhotonCounting.distributions.PoissonPhotonLikelihood()
noise_likelihood = GWPhotonCounting.distributions.PhaseQuadraturePhotonLikelihood()
mixture_likelihood = GWPhotonCounting.distributions.MixturePhotonLikelihood(poisson_likelihood, noise_likelihood)
gaussian_likelihood = GWPhotonCounting.distributions.GaussianStrainLikelihood()

snr_factor = 1/detector_nosqz.calculate_optimal_snr(PM_strain, frequencies, fmin=1.5e3)

print(snr_factor)

for i, fac in enumerate([5]):

    PM_strain_i = PM_strain * fac * snr_factor

    expected_signal_photon_count = detector_nosqz.calculate_signal_photon_expectation(PM_strain_i, frequencies)

    print('Optimal SNR', detector_nosqz.calculate_optimal_snr(PM_strain_i, frequencies, fmin=1.5e3))
    print('Optimal sqz SNR', detector_sqz.calculate_optimal_snr(PM_strain_i, frequencies, fmin=1.5e3))
    print('Expected photon count from SNR', detector_nosqz.calculate_optimal_snr(PM_strain_i, frequencies, fmin=1.5e3)**2/2)
    print('Expected photon count from filters', jnp.sum(expected_signal_photon_count))

    # Simulating a photon count
    observed_photons, signal_photons, noise_photons = mixture_likelihood.generate_realization(
        detector_nosqz.calculate_signal_photon_expectation(PM_strain_i, frequencies), 
        detector_nosqz.noise_photon_expectation)
    
    print(jnp.sum(signal_photons), jnp.sum(noise_photons))
    if jnp.sum(signal_photons) == 0:
        signal_photons = signal_photons.at[jnp.argmax(expected_signal_photon_count)].set(1)
        print('Probability of signal photon being 1:', 1-jnp.exp(poisson_likelihood.log_likelihood(jnp.zeros(200), expected_signal_photon_count)))


    fit_lorentzian_n = GWPhotonCounting.inference.PhotonCountingInference(detector_nosqz, frequencies, mixture_likelihood).run(
        signal_photons,
        num_warmup=2000,
        num_samples=4000,
        num_chains=2, time_reconstruction=True, f0min=1.5e3)
    
    fit_lorentzian_n.to_netcdf('result_high_low_snr_250615c/pc_snr_scaling_fac{}.nc'.format(fac))

    fit_lorentzian_strain = GWPhotonCounting.inference.StrainInference(detector_nosqz, frequencies, gaussian_likelihood).run(
        PM_strain_i,
        num_warmup=2000,
        num_samples=4000,
        num_chains=2, time_reconstruction=True, f0min=1.5e3)
    
    fit_lorentzian_strain.to_netcdf('result_high_low_snr_250615c/strain_snr_scaling_fac{}.nc'.format(fac))
