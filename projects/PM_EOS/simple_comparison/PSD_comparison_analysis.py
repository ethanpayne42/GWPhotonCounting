import GWPhotonCounting
import jax.numpy as jnp

frequencies = jnp.sort(jnp.fft.fftfreq(2**13, d=1/1e4))

detector_nosqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_shot_psd_nosqz.csv', 
    '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_classical_psd.csv', 
    gamma=100, random_seed=1632, N_frequency_spaces=10)

LorentzianModel = GWPhotonCounting.signal.PostMergerLorentzian()
KNNModel = GWPhotonCounting.signal.PostMergerKNN(knn_file_path='/home/ethan.payne/code_libraries/apr4_knn_gw_model_2024/KNN_Models/APR4-knn_model-N100')

# Generating the strain signal
from bilby_cython.geometry import frequency_dependent_detector_tensor
import numpy as np

mtot = 2.5
phi0 = 0
z = 0.03
ra = 0.0
dec = 0.3
iota=0.2
psi = 0

PM_strain = KNNModel.generate_strain(detector_nosqz, frequencies, mtot, phi0, z, ra, dec, iota, psi)

print(jnp.max(np.abs(PM_strain)))

expected_signal_photon_count = detector_nosqz.calculate_signal_photon_expectation(PM_strain, frequencies)

print('Optimal SNR', detector_nosqz.calculate_optimal_snr(PM_strain, frequencies))
print('Expected photon count from SNR', detector_nosqz.calculate_optimal_snr(PM_strain, frequencies)**2/2)
print('Expected photon count from filters', jnp.sum(expected_signal_photon_count))

poisson_likelihood = GWPhotonCounting.distributions.PoissonPhotonLikelihood()
noise_likelihood = GWPhotonCounting.distributions.PhaseQuadraturePhotonLikelihood()
mixture_likelihood = GWPhotonCounting.distributions.MixturePhotonLikelihood(poisson_likelihood, noise_likelihood)
gaussian_likelihood = GWPhotonCounting.distributions.GaussianStrainLikelihood()

# Simulating a photon count
observed_signal_photons = jnp.zeros(len(expected_signal_photon_count))
observed_signal_photons = observed_signal_photons.at[jnp.argmax(expected_signal_photon_count)].set(1)
observed_noise_photons = jnp.zeros(len(expected_signal_photon_count))
observed_photons = observed_signal_photons + observed_noise_photons

for i, noise_scale in enumerate(jnp.logspace(-2, 2, 9)):

    fit_lorentzian_n = GWPhotonCounting.inference.PhotonCountingInference(detector_nosqz, frequencies, mixture_likelihood).run(
        observed_photons,
        num_warmup=1000,
        num_samples=1000,
        num_chains=2, time_reconstruction=False, f0min=1.5e3,
        noise_scale=noise_scale)
    
    fit_lorentzian_n.to_netcdf('result_noise_scaling_250418/pc_noise_scale_{}.nc'.format(i))

for i, noise_scale in enumerate(10**(-jnp.linspace(10,20,9)/10)):

    fit_lorentzian_strain = GWPhotonCounting.inference.StrainInference(detector_nosqz, frequencies, gaussian_likelihood).run(
        PM_strain,
        num_warmup=1000,
        num_samples=1000,
        num_chains=2, time_reconstruction=False, f0min=1.5e3,
        noise_scale=noise_scale)
    
    fit_lorentzian_strain.to_netcdf('result_noise_scaling_250418/strain_noise_scale_{}.nc'.format(i))
