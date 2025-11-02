import GWPhotonCounting
import jax.numpy as jnp

frequencies = jnp.sort(jnp.fft.fftfreq(2**13, d=1/1e4))

detector = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_shot_psd_nosqz.csv', 
    '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE_classical_psd.csv', 
    gamma=100, random_seed=1632, N_frequency_spaces=10)

LorentzianModel = GWPhotonCounting.signal.PostMergerLorentzian()
KNNModel = GWPhotonCounting.signal.PostMergerKNN(knn_file_path='/home/ethan.payne/code_libraries/apr4_knn_gw_model_2024/KNN_Models/APR4-knn_model-N100')

# Generating the strain signal
from bilby_cython.geometry import frequency_dependent_detector_tensor
import numpy as np

mtot = 2.4
phi0 = 0
z = 0.05
ra = jnp.pi/4
dec = jnp.pi/4
iota=jnp.pi/4
psi = 0

PM_strain = KNNModel.generate_strain(detector, frequencies, mtot, phi0, z, ra, dec, iota, psi)

print(jnp.max(np.abs(PM_strain)))

expected_signal_photon_count = detector.calculate_signal_photon_expectation(PM_strain, frequencies)

print(expected_signal_photon_count)


poisson_likelihood = GWPhotonCounting.distributions.PoissonPhotonLikelihood()
noise_likelihood = GWPhotonCounting.distributions.PhaseQuadraturePhotonLikelihood()
mixture_likelihood = GWPhotonCounting.distributions.MixturePhotonLikelihood(poisson_likelihood, noise_likelihood)
gaussian_likelihood = GWPhotonCounting.distributions.GaussianStrainLikelihood()

for i, n in enumerate(np.linspace(0,3,4, dtype=int)):

    scaled_expected_signal_photon_count = expected_signal_photon_count * n / np.sum(expected_signal_photon_count)
    observed_signal_photons = poisson_likelihood.generate_realization(scaled_expected_signal_photon_count)

    while np.sum(observed_signal_photons) != n:
        observed_signal_photons = poisson_likelihood.generate_realization(scaled_expected_signal_photon_count)

    print(observed_signal_photons[np.argwhere(observed_signal_photons > 0)])
    print(expected_signal_photon_count[np.argwhere(observed_signal_photons > 0)])

    fit_lorentzian_n = GWPhotonCounting.inference.PhotonCountingInference(detector, frequencies, mixture_likelihood).run(
        observed_signal_photons * n,
        num_warmup=1000,
        num_samples=1000,
        num_chains=2, time_reconstruction=True, f0min=1.5e3)
    
    fit_lorentzian_n.to_netcdf(f'result_photon_scaling_250608d/pc_{int(n)}.nc'.format(i))