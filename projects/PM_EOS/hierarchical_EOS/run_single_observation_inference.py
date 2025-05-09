import GWPhotonCounting
import jax.numpy as jnp
import numpy as np
import bilby
from bilby_cython.geometry import frequency_dependent_detector_tensor

import json
import sys
import os

from astropy.cosmology import Planck18
import astropy.units as u

# injected snr
idx = int(sys.argv[1])

frequencies = jnp.sort(jnp.fft.fftfreq(2**13, d=1/1e4))

# Setting up the two detectors to compare the 
detector_CE1 = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE1_shot_psd_updated.csv', 
    '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE1_classical_quanta_updated.csv', 
    gamma=100, random_seed=1632, N_frequency_spaces=10)

detector_CE2silica = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE2silica_sqz_total_psd.csv', None,
    gamma=100, random_seed=1632, N_frequency_spaces=10)

detector_CE2silicon = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE2silicon_shot_psd.csv', 
    '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE2silicon_classical_quanta.csv',
    gamma=100, random_seed=1632, N_frequency_spaces=10)

detector_CE2silicon_sqz = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE2silicon_sqz_total_psd.csv', None,
    gamma=100, random_seed=1632, N_frequency_spaces=10)

#Loading in the individual analysis from the sample
KNNModel = GWPhotonCounting.signal.PostMergerKNN(knn_file_path='/home/ethan.payne/code_libraries/apr4_knn_gw_model_2024/KNN_Models/APR4-knn_model-N100')
dataset = np.genfromtxt(f'/home/ethan.payne/projects/GWPhotonCounting/projects/PM_EOS/hierarchical_EOS/bns_pm_dataset_updated.dat')

mtots, z, phi, psi, ra, dec, iota, fpeak_true, snr, snr_silica, snr_silicon = dataset[idx]
print('SNRs are: ', snr, snr_silica, snr_silicon)

# Compute the expected number of photons and the strain
PM_strain = KNNModel.generate_strain(detector_CE1, frequencies, mtots, phi, z, ra, dec, iota, psi)

# Setting up the likelihood
poisson_likelihood = GWPhotonCounting.distributions.PoissonPhotonLikelihood()
noise_likelihood = GWPhotonCounting.distributions.PhaseQuadraturePhotonLikelihood() 
gaussian_likelihood = GWPhotonCounting.distributions.GaussianStrainLikelihood()
# Use GWPhotonCounting.distributions.GeometricPhotonLikelihood() if you want a geometric distribution

# Setting up the convolution of the noise and poisson distributions
convolved_likelihood = GWPhotonCounting.distributions.MixturePhotonLikelihood(poisson_likelihood, noise_likelihood)

# Compute the min and max frequencies
f0_bounds = GWPhotonCounting.hierarchical.frequency_model(mtots, jnp.array([17,7.5]))/(1+z)
print(f0_bounds)

# Calculation for the CE1 detector
observed_photons, signal_photons, _ = convolved_likelihood.generate_realization(detector_CE1.calculate_signal_photon_expectation(PM_strain, frequencies), detector_CE1.noise_photon_expectation)
observed_photons_no_background = poisson_likelihood.generate_realization(detector_CE1.calculate_signal_photon_expectation(PM_strain, frequencies))
observed_strain = PM_strain + gaussian_likelihood.generate_realization(detector_CE2silica.total_psd, frequencies)

fit_lorentzian_n_no_background = None
if jnp.sum(observed_photons_no_background) > 0:
    print(f'Poisson background (CE1). Photons detected: {jnp.sum(observed_photons_no_background)} (signal: {jnp.sum(observed_photons_no_background)})')
    fit_lorentzian_n_no_background = GWPhotonCounting.inference.PhotonCountingInference(detector_CE1, frequencies, poisson_likelihood, include_background=False).run(
        observed_photons_no_background, time_reconstruction=False,
        num_warmup=1000,
        num_samples=1000,
        num_chains=5,
        f0min=f0_bounds[0], 
        f0max=f0_bounds[1])

fit_lorentzian_n = None
if jnp.sum(observed_photons) > 0:
    print(f'Background (CE1). Photons detected: {jnp.sum(observed_photons)} (signal: {jnp.sum(signal_photons)})')
    fit_lorentzian_n = GWPhotonCounting.inference.PhotonCountingInference(detector_CE1, frequencies, convolved_likelihood).run(
        observed_photons, time_reconstruction=False,
        num_warmup=1000,
        num_samples=1000,
        num_chains=5,
        f0min=f0_bounds[0], 
        f0max=f0_bounds[1])


fit_lorentzian_strain = GWPhotonCounting.inference.StrainInference(detector_CE2silica, frequencies, gaussian_likelihood).run(
    observed_strain, time_reconstruction=False,
    num_warmup=1000,
    num_samples=1000,
    num_chains=5,
    f0min=f0_bounds[0], 
    f0max=f0_bounds[1])

GWPhotonCounting.inference.save_analyses(
    f'result_{idx}', fit_pc=fit_lorentzian_n, fit_pc_no_background=fit_lorentzian_n_no_background, 
    fit_strain=fit_lorentzian_strain, outdir='results_250403b_CE1CE2silica/',
    mtot=mtots, z=z, phi=phi, psi=psi, ra=ra, dec=dec, iota=iota, fpeak_true=fpeak_true, snr=snr, snr_squeeze=snr_silica,
    n_photons=float(jnp.sum(observed_photons)), n_signal_photons=float(jnp.sum(signal_photons)),
    n_photons_no_background=float(jnp.sum(observed_photons_no_background)))


# Calculation for the CE2silicon detector
observed_photons, signal_photons, _ = convolved_likelihood.generate_realization(detector_CE2silicon.calculate_signal_photon_expectation(PM_strain, frequencies), detector_CE2silicon.noise_photon_expectation)
observed_photons_no_background = poisson_likelihood.generate_realization(detector_CE2silicon.calculate_signal_photon_expectation(PM_strain, frequencies))
observed_strain = PM_strain + gaussian_likelihood.generate_realization(detector_CE2silicon_sqz.total_psd, frequencies)

fit_lorentzian_n_no_background = None
if jnp.sum(observed_photons_no_background) > 0:
    print(f'Poisson background. Photons detected: {jnp.sum(observed_photons_no_background)} (signal: {jnp.sum(observed_photons_no_background)})')
    fit_lorentzian_n_no_background = GWPhotonCounting.inference.PhotonCountingInference(detector_CE2silicon, frequencies, poisson_likelihood, include_background=False).run(
        observed_photons_no_background, time_reconstruction=False,
        num_warmup=1000,
        num_samples=1000,
        num_chains=5,
        f0min=f0_bounds[0], 
        f0max=f0_bounds[1])

fit_lorentzian_n = None
if jnp.sum(observed_photons) > 0:
    print(f'Background CE2silicon. Photons detected: {jnp.sum(observed_photons)} (signal: {jnp.sum(signal_photons)})')
    fit_lorentzian_n = GWPhotonCounting.inference.PhotonCountingInference(detector_CE2silicon, frequencies, convolved_likelihood).run(
        observed_photons, time_reconstruction=False,
        num_warmup=1000,
        num_samples=1000,
        num_chains=5,
        f0min=f0_bounds[0], 
        f0max=f0_bounds[1])

    
fit_lorentzian_strain = GWPhotonCounting.inference.StrainInference(detector_CE2silicon_sqz, frequencies, gaussian_likelihood).run(
    observed_strain, time_reconstruction=False,
    num_warmup=1000,
    num_samples=1000,
    num_chains=5,
    f0min=f0_bounds[0], 
    f0max=f0_bounds[1])

GWPhotonCounting.inference.save_analyses(
    f'result_{idx}', fit_pc=fit_lorentzian_n, fit_pc_no_background=fit_lorentzian_n_no_background, 
    fit_strain=fit_lorentzian_strain, outdir='results_250403b_CE2silicon/',
    mtot=mtots, z=z, phi=phi, psi=psi, ra=ra, dec=dec, iota=iota, fpeak_true=fpeak_true, snr=snr_silicon, snr_squeeze=snr_silicon,
    n_photons=float(jnp.sum(observed_photons)), n_signal_photons=float(jnp.sum(signal_photons)),
    n_photons_no_background=float(jnp.sum(observed_photons_no_background)))