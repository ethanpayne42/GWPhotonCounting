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
mtot = 2.5
phi0 = 0
z = 0.03
ra = 0.0
dec = 0.3
iota=0.2
psi = 0

PM_strain = KNNModel.generate_strain(detector_nosqz, frequencies, mtot, phi0, z, ra, dec, iota, psi)

expected_signal_photon_count = detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies)

poisson_likelihood = GWPhotonCounting.distributions.PoissonPhotonLikelihood()
noise_likelihood = GWPhotonCounting.distributions.PhaseQuadraturePhotonLikelihood()
mixture_likelihood = GWPhotonCounting.distributions.MixturePhotonLikelihood(poisson_likelihood, noise_likelihood)
gaussian_likelihood = GWPhotonCounting.distributions.GaussianStrainLikelihood()

snr_factor = 0.2/detector_nosqz.calculate_optimal_snr(PM_strain, frequencies)

for i, fac in enumerate([1, 13]):

    PM_strain_i = PM_strain * fac * snr_factor

    expected_signal_photon_count = detector_nosqz.calculate_signal_photon_expectation(PM_strain_i, frequencies)

    print('Optimal SNR', detector_nosqz.calculate_optimal_snr(PM_strain_i, frequencies))
    print('Optimal sqz SNR', detector_sqz.calculate_optimal_snr(PM_strain_i, frequencies))
    print('Expected photon count from SNR', detector_nosqz.calculate_optimal_snr(PM_strain_i, frequencies)**2/2)
    print('Expected photon count from filters', jnp.sum(expected_signal_photon_count))

    # Simulating a photon count
    observed_signal_photons = jnp.zeros(len(expected_signal_photon_count))
    observed_signal_photons = observed_signal_photons.at[jnp.argmax(expected_signal_photon_count)].set(1)
    observed_noise_photons = jnp.zeros(len(expected_signal_photon_count))
    observed_photons = observed_signal_photons + observed_noise_photons


    fit_lorentzian_n = GWPhotonCounting.inference.PhotonCountingInference(detector_nosqz, frequencies, mixture_likelihood).run(
        observed_photons,
        num_warmup=1000,
        num_samples=1000,
        num_chains=2, time_reconstruction=False, f0min=1.5e3)
    
    fit_lorentzian_n.to_netcdf('result_high_low_snr_250422f/pc_snr_scaling_{}.nc'.format(i))

    fit_lorentzian_strain = GWPhotonCounting.inference.StrainInference(detector_sqz, frequencies, gaussian_likelihood).run(
        PM_strain_i,
        num_warmup=1000,
        num_samples=1000,
        num_chains=2, time_reconstruction=False, f0min=1.5e3)
    
    fit_lorentzian_strain.to_netcdf('result_high_low_snr_250422f/strain_snr_scaling_{}.nc'.format(i))
