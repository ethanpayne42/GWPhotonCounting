

from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Planck18
from bilby_cython.geometry import frequency_dependent_detector_tensor
from jax.scipy.special import logsumexp

import arviz as az
import os
from tqdm import tqdm
import json
import glob

import matplotlib.pyplot as plt

def frequency_model(mtot, R1d6):

        mchirp = mtot * 0.435 # assuming equal mass ratio

        beta0 = 1.5220
        beta1 = 8.4021
        beta2 = 2.3876
        beta3 = -1.1133
        beta4 = -0.1291
        beta5 = 0.0366

        return (beta0 + beta1 * mchirp + beta2 * mchirp**2 + beta3 * R1d6 * mchirp + beta4 * R1d6 * mchirp**2 + beta5 * R1d6**2 * mchirp)*1e3


def invert_frequency_model(mtot, frequency):
    # Constants
    beta0 = 1.5220
    beta1 = 8.4021
    beta2 = 2.3876
    beta3 = -1.1133
    beta4 = -0.1291
    beta5 = 0.0366

    # Compute mchirp
    mchirp = mtot * 0.435  # equal mass assumption

    # Coefficients for the quadratic equation A*x^2 + B*x + C = 0
    A = beta5 * mchirp
    B = beta3 * mchirp + beta4 * mchirp**2
    C = beta0 + beta1 * mchirp + beta2 * mchirp**2 - frequency / 1e3

    # Discriminant
    discriminant = B**2 - 4*A*C

    # Solve using quadratic formula
    sqrt_disc = jnp.sqrt(discriminant)
    R1d6_1 = (-B + sqrt_disc) / (2*A)
    R1d6_2 = (-B - sqrt_disc) / (2*A)

    return R1d6_2

def frequency_model_derivative(mtot, R1d6):

        mchirp = mtot * 0.435 # assuming equal mass ratio

        beta0 = 1.5220
        beta1 = 8.4021
        beta2 = 2.3876
        beta3 = -1.1133
        beta4 = -0.1291
        beta5 = 0.0366

        return jnp.abs((beta3 * mchirp + beta4 * mchirp**2 + 2*beta5 * R1d6 * mchirp)*1e3)

def calculate_likelihoods(f0s, mtot, z, noise_only_loglikelihood): # f0s, frequencies
    
    # f0_min = np.min(f0s)
    # f0_max = np.max(f0s)

    # f0s_new = np.sort(f0s)
    # f0s_new = f0s_new[np.abs(np.diff(f0s_new, append=jnp.array(f0s_new[-1]))) > 0]

    # # concat_array = np.concatenate([f0s_new, 2*f0_min - f0s_new, 2*f0_max - f0s_new])

    # # density = gaussian_kde(concat_array,bw_method=0.05)(frequencies) #f0s 

    # bin_density, bin_edges = np.histogram(f0s_new, bins=np.linspace(f0_min, f0_max, 20), density=True)

    # density = interp1d(bin_edges[:-1], np.log(bin_density), bounds_error=False, fill_value=0, kind='linear')(frequencies)

    R1d6_samples = invert_frequency_model(mtot, f0s*(1+z))
    good_indexes = jnp.logical_and(R1d6_samples > 7.5, R1d6_samples < 17)
    R1d6_samples = R1d6_samples[good_indexes]
    weight_noise = interp1d(jnp.linspace(1e2, 4e3, 100)[:-1], np.exp(noise_only_loglikelihood), kind='linear', fill_value='extrapolate')(f0s[good_indexes])
    total_weight_factor = 1/frequency_model_derivative(mtot, R1d6_samples)/weight_noise

    concat_array = np.concatenate([R1d6_samples, 2*7.5 - R1d6_samples, 2*17 - R1d6_samples])
    concat_weights = np.concatenate([total_weight_factor, total_weight_factor, total_weight_factor])


    density = np.histogram(concat_array, bins=np.linspace(7.5,17,30), density=True, weights=concat_weights)[0]


    return jnp.array(density)

class HierarchicalPostMerger():
    def __init__(self, results_directory, detector, Nevents=1000, inspiral_uncertainty=False,
                 noise_directory='/home/ethan.payne/projects/GWPhotonCounting/projects/PM_EOS/hierarchical_EOS/results_noise_only_250401_CE1CE2silica/'):
        
        self.results_directory = results_directory
        self.noise_directory = noise_directory
        self.inspiral_uncertainty = inspiral_uncertainty
        self.N_events = Nevents
        self.detector = detector

        self.mtots, self.zs, self.snrs, self.snr_squeezes, self.likelihood_photon, self.likelihood_photon_no_background, self.likelihood_strain = self._load_results()

        self.N_events = len(self.mtots)

    def _load_results(
            self,
            photon_no_detection='/home/ethan.payne/projects/GWPhotonCounting/projects/PM_EOS/hierarchical_EOS/results_250401_CE1CE2silica/no_detection_pc.nc', 
            photon_no_detection_no_background='/home/ethan.payne/projects/GWPhotonCounting/projects/PM_EOS/hierarchical_EOS/results_250401_CE1CE2silica/no_detection_pc_no_background.nc'):

        mtots = []
        zs = []
        snrs = []
        snr_squeezes = []
        likelihood_photon = []
        likelihood_photon_no_background = []
        likelihood_strain = []

        frequencies = jnp.linspace(1e2, 4e3, 100)
        R1d6s = jnp.linspace(9, 15, 100)


        N_events_count = 0

        self.photon_count = 0
        self.signal_photon_count = 0
        self.photon_count_no_background = 0

        likelihood_noise_only_pcs = []
        likelihood_noise_only_strains = []

    

        likelihood_no_detection_pc = jnp.log(np.histogram(az.from_netcdf(photon_no_detection).posterior.f0.values.flatten(), bins=frequencies, density=True)[0])

        for i in tqdm(range(1000)):
            try:
                likelihood_noise_only_strains.append(jnp.log(np.histogram(az.from_netcdf(self.noise_directory + f'/result_{i}_strain.nc').posterior.f0.values.flatten(), bins=frequencies, density=True)[0]))

                if os.path.exists(self.noise_directory + f'/result_{i}_pc.nc'):
                    data = az.from_netcdf(self.noise_directory + f'/result_{i}_pc.nc')
                    likelihood_noise_only_pcs.append(jnp.log(np.histogram(data.posterior.f0.values.flatten(), bins=frequencies, density=True)[0]))

                else:
                    likelihood_noise_only_pcs.append(likelihood_no_detection_pc)
            except:
                pass

        self.likelihood_noise_only_pc = logsumexp(jnp.array(likelihood_noise_only_pcs),axis=0) - jnp.log(1000)
        self.likelihood_noise_only_pc_no_background = logsumexp(jnp.array(likelihood_noise_only_pcs),axis=0) - jnp.log(1000)
        self.likelihood_noise_only_strain = logsumexp(jnp.array(likelihood_noise_only_strains),axis=0) - jnp.log(1000)

        pc_no_detection_f0s = az.from_netcdf(photon_no_detection).posterior.f0.values.flatten()
        pc_no_detection_no_background_f0s = az.from_netcdf(photon_no_detection_no_background).posterior.f0.values.flatten()

        for i in tqdm(range(1000, 1000+self.N_events)):
            file = f'result_{i}.json'
            with open(self.results_directory + '/' + file, 'r') as f:
                data_json = json.load(f)
                
                mtots.append(data_json['mtot'])
                zs.append(data_json['z'])
                snrs.append(data_json['snr'])
                snr_squeezes.append(data_json['snr_squeeze'])                

                if 'filename_pc' in data_json:
                    data = az.from_netcdf(data_json['filename_pc'])
                    likelihood_photon.append(jnp.log(calculate_likelihoods(data.posterior.f0.values.flatten(), 
                                                                           data_json['mtot'], data_json['z'], self.likelihood_noise_only_pc)))
                    self.photon_count += int(data_json['n_photons'])
                    self.signal_photon_count += int(data_json['n_signal_photons'])

                else:
                    likelihood_no_detection_pc = jnp.log(calculate_likelihoods(pc_no_detection_f0s, 
                                                                           data_json['mtot'], data_json['z'], self.likelihood_noise_only_pc))
                    
                    likelihood_photon.append(likelihood_no_detection_pc)

                if 'filename_pc_no_background' in data_json:
                    data = az.from_netcdf(data_json['filename_pc_no_background'])
                    likelihood_photon_no_background.append(jnp.log(calculate_likelihoods(data.posterior.f0.values.flatten(), 
                                                                                         data_json['mtot'], data_json['z'], self.likelihood_noise_only_pc_no_background)))
                    self.photon_count_no_background += int(data_json['n_photons_no_background'])

                else:
                    likelihood_no_detection_pc_no_background = jnp.log(calculate_likelihoods(pc_no_detection_no_background_f0s, 
                                                                                         data_json['mtot'], data_json['z'], self.likelihood_noise_only_pc_no_background))
                    likelihood_photon_no_background.append(likelihood_no_detection_pc_no_background)


                data = az.from_netcdf(data_json['filename_strain'])
                likelihood_strain.append(jnp.log(calculate_likelihoods(data.posterior.f0.values.flatten(), data_json['mtot'], data_json['z'], self.likelihood_noise_only_strain)))
                N_events_count += 1

        return jnp.array(mtots), jnp.array(zs), jnp.array(snrs), jnp.array(snr_squeezes),jnp.array(likelihood_photon), \
               jnp.array(likelihood_photon_no_background), jnp.array(likelihood_strain)
    
    
    def generate_posterior(self, N_total=None):
        
        R1d6s = jnp.linspace(7.5, 17, 30)

        if N_total is None:
            N_total = self.N_events

        if not self.inspiral_uncertainty:

            hier_loglikelihood_photon = jnp.sum(self.likelihood_photon[:N_total], axis=0)
            hier_loglikelihood_photon_no_background = jnp.sum(self.likelihood_photon_no_background[:N_total], axis=0)
            hier_loglikelihood_strain = jnp.sum(self.likelihood_strain[:N_total], axis=0)

        else:
            raise ValueError('TODO NOT IMPLEMENTED YET')
        
        return R1d6s, hier_loglikelihood_photon, hier_loglikelihood_photon_no_background, hier_loglikelihood_strain





            

    

    