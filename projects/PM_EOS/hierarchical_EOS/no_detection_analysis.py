import GWPhotonCounting
import jax.numpy as jnp
import numpy as np

from corner import corner
import matplotlib
from matplotlib.lines import Line2D

frequencies = jnp.sort(jnp.fft.fftfreq(2**13, d=1/1e4))
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

# Setting up the likelihood
poisson_likelihood = GWPhotonCounting.distributions.PoissonPhotonLikelihood()
noise_likelihood = GWPhotonCounting.distributions.PhaseQuadraturePhotonLikelihood() 
# Use GWPhotonCounting.distributions.GeometricPhotonLikelihood() if you want a geometric distribution

# Setting up the convolution of the noise and poisson distributions
convolved_likelihood = GWPhotonCounting.distributions.MixturePhotonLikelihood(poisson_likelihood, noise_likelihood)

observed_photons = jnp.zeros(detector_CE1.N_total_filters)


fit_lorentzian_n = GWPhotonCounting.inference.PhotonCountingInference(detector_CE1, frequencies, convolved_likelihood).run(
    observed_photons,
    num_warmup=1000,
    num_samples=2000,
    num_chains=10)

fit_lorentzian_n_no_background = GWPhotonCounting.inference.PhotonCountingInference(detector_CE2silica, frequencies, poisson_likelihood, include_background=False).run(
    observed_photons,
    num_warmup=1000,
    num_samples=2000,
    num_chains=10)

GWPhotonCounting.inference.save_analyses(f'no_detection', fit_pc=fit_lorentzian_n, fit_pc_no_background=fit_lorentzian_n_no_background, outdir='results_250401_CE1CE2silica/')

fit_lorentzian_n = GWPhotonCounting.inference.PhotonCountingInference(detector_CE2silicon, frequencies, convolved_likelihood).run(
    observed_photons,
    num_warmup=1000,
    num_samples=2000,
    num_chains=10)

fit_lorentzian_n_no_background = GWPhotonCounting.inference.PhotonCountingInference(detector_CE2silicon_sqz, frequencies, poisson_likelihood, include_background=False).run(
    observed_photons,
    num_warmup=1000,
    num_samples=2000,
    num_chains=10)

GWPhotonCounting.inference.save_analyses(f'no_detection', fit_pc=fit_lorentzian_n, fit_pc_no_background=fit_lorentzian_n_no_background, outdir='results_250401_CE2silicon/')