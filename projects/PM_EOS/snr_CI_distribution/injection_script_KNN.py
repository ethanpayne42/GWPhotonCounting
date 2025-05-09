import GWPhotonCounting
import jax.numpy as jnp
import numpy as np
import bilby

import sys
import os

from astropy.cosmology import Planck18
import astropy.units as u

# injected snr
snr_inj = float(sys.argv[1])
idx = int(sys.argv[2])

frequencies = jnp.sort(jnp.fft.fftfreq(2**13, d=1/1e4))

# Setting up the two detectors to compare the 
detector = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE1_shot_psd_updated.csv', 
    '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE1_classical_quanta_updated.csv', 
    gamma=100, random_seed=1632, N_frequency_spaces=10)

zmax = 10
zinterp = np.expm1(np.linspace(np.log1p(0), np.log1p(zmax), 2000))
dVdzdt_interp = 4*np.pi*Planck18.differential_comoving_volume(zinterp).to(u.Gpc**3/u.sr).value/(1+zinterp)

KNNModel = GWPhotonCounting.signal.PostMergerKNN(knn_file_path='/home/ethan.payne/code_libraries/apr4_knn_gw_model_2024/KNN_Models/APR4-knn_model-N100')

def sample_redshift(n):
    pdf_red = ((1+zinterp)**2.7)/(1+((1+zinterp)/2.9)**5.6) * dVdzdt_interp
    cum_sum_red = np.cumsum(pdf_red)/np.sum(pdf_red)
    
    return np.interp(np.random.uniform(size=n), cum_sum_red, zinterp)

fpeak_true = 0
while fpeak_true < 1.8e3:
    m1 = bilby.gw.prior.Uniform(1.2,1.4).sample(1)
    m2 = bilby.gw.prior.Uniform(1.2,1.4).sample(1)
    mtots = m1+m2

    z = sample_redshift(1)

    phi = bilby.gw.prior.Uniform(0,2*np.pi).sample(1)
    psi = bilby.gw.prior.Uniform(0,np.pi).sample(1)

    ra = bilby.core.prior.Uniform(0,2*np.pi).sample(1)
    dec = bilby.core.prior.Cosine().sample(1)

    iota = bilby.core.prior.Sine().sample(1)
    
    PM_strain = KNNModel.generate_strain(detector, frequencies, mtots, phi, z, ra, dec, iota, psi)

    fpeak_true = np.abs(frequencies[np.argmax(np.abs(PM_strain))])

snr = detector.calculate_optimal_snr(PM_strain, frequencies)

# Scale the strain to the desired SNR and generate the photon count expectation
PM_strain_scaled = PM_strain * snr_inj/snr
expected_signal_photon_count = detector.calculate_signal_photon_expectation(PM_strain_scaled, frequencies)

# Setting up the likelihood
poisson_likelihood = GWPhotonCounting.distributions.PoissonPhotonLikelihood()
noise_likelihood = GWPhotonCounting.distributions.PhaseQuadraturePhotonLikelihood() 
gaussian_likelihood = GWPhotonCounting.distributions.GaussianStrainLikelihood()
# Use GWPhotonCounting.distributions.GeometricPhotonLikelihood() if you want a geometric distribution

# Setting up the convolution of the noise and poisson distributions
convolved_likelihood = GWPhotonCounting.distributions.MixturePhotonLikelihood(poisson_likelihood, noise_likelihood)

# Generate the observed signals from the likelihoods
observed_photons = poisson_likelihood.generate_realization(expected_signal_photon_count)
observed_strain = PM_strain_scaled + gaussian_likelihood.generate_realization(detector.total_psd, frequencies)

# Run the inference calculations
fit_lorentzian_n = GWPhotonCounting.inference.PhotonCountingInference(detector, frequencies, convolved_likelihood).run(
    observed_photons, num_chains=2, f0min=1.5e3, time_reconstruction=False)

CI_pc = np.diff(np.quantile(fit_lorentzian_n.posterior.f0.values.flatten(), np.array([0.16,0.84])))[0]

fit_lorentzian_n_no_background = GWPhotonCounting.inference.PhotonCountingInference(detector, frequencies, poisson_likelihood, include_background=False).run(
    observed_photons, num_chains=2, f0min=1.5e3, time_reconstruction=False)

CI_pc_no_background = np.diff(np.quantile(fit_lorentzian_n_no_background.posterior.f0.values.flatten(), np.array([0.16,0.84])))[0]

fit_lorentzian_strain = GWPhotonCounting.inference.StrainInference(detector, frequencies, gaussian_likelihood).run(
    observed_strain, num_chains=2, f0min=1.5e3, time_reconstruction=False)

CI_strain = np.diff(np.quantile(fit_lorentzian_strain.posterior.f0.values.flatten(), np.array([0.16,0.84])))[0]

if not os.path.exists('results'):
    os.makedirs('results')

np.savetxt(f'results_250311/summary_{snr_inj}_{idx}.dat', np.array([float(snr_inj), CI_pc, CI_pc_no_background, CI_strain]))


    


