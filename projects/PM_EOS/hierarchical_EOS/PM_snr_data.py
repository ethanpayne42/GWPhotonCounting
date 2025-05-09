import GWPhotonCounting
import jax.numpy as jnp
import numpy as np
import bilby

from corner import corner
import matplotlib.pyplot as plt
from tqdm import tqdm

from astropy.cosmology import Planck18
import astropy.units as u

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
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE2silicon_sqz_total_psd.csv', None,
    gamma=100, random_seed=1632, N_frequency_spaces=10)

zmax = 10
zinterp = np.expm1(np.linspace(np.log1p(0), np.log1p(zmax), 2000))
dVdzdt_interp = 4*np.pi*Planck18.differential_comoving_volume(zinterp).to(u.Gpc**3/u.sr).value/(1+zinterp)

KNNModel = GWPhotonCounting.signal.PostMergerKNN(knn_file_path='/home/ethan.payne/code_libraries/apr4_knn_gw_model_2024/KNN_Models/APR4-knn_model-N100')

Vc = np.sum((((1+zinterp)**2.7)/(1+((1+zinterp)/2.9)**5.6) /(((1)**2.7)/(1+((1)/2.9)**5.6)) * 4*np.pi*Planck18.differential_comoving_volume(zinterp).to(u.Gpc**3/u.sr).value/(1+zinterp))[:-1] * np.diff(zinterp))
print(Vc)

def sample_redshift(n):
    pdf_red = ((1+zinterp)**2.7)/(1+((1+zinterp)/2.9)**5.6) * dVdzdt_interp
    cum_sum_red = np.cumsum(pdf_red)/np.sum(pdf_red)
    
    return np.interp(np.random.uniform(size=n), cum_sum_red, zinterp)

rate = np.array([320-240,320,320+490]) # Gpc^-3 yr^-1
total_comoving_volume = Vc
observing_time = 1 # yr
total_expected_events = rate * total_comoving_volume * observing_time
print(f"Total expected events: {total_expected_events}")

# Adding samples to the dataset
snr_array = []

for N in tqdm(range(int(total_expected_events[1]))):
    m1 = bilby.gw.prior.Uniform(1.2,1.4).sample(1)[0]
    m2 = bilby.gw.prior.Uniform(1.2,1.4).sample(1)[0]
    mtots = m1+m2

    z = sample_redshift(1)[0]

    phi = bilby.gw.prior.Uniform(0,2*np.pi).sample(1)[0]
    psi = bilby.gw.prior.Uniform(0,np.pi).sample(1)[0]

    ra = bilby.core.prior.Uniform(0,2*np.pi).sample(1)[0]
    dec = bilby.core.prior.Cosine().sample(1)[0]

    iota = bilby.core.prior.Sine().sample(1)[0]
    
    PM_strain = KNNModel.generate_strain(detector_CE1, frequencies, mtots, phi, z, ra, dec, iota, psi)

    fpeak_true = np.abs(frequencies[np.argmax(np.abs(PM_strain))])

    SNR_CE2silica = detector_CE2silica.calculate_optimal_snr(PM_strain, frequencies)

    snr_array.append(SNR_CE2silica)

np.save('PM_snrs_full.npy', snr_array)