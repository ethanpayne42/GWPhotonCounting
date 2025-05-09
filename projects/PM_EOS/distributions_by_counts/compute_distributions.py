import GWPhotonCounting
import jax.numpy as jnp
import numpy as np

from corner import corner
import matplotlib
from matplotlib.lines import Line2D

frequencies = jnp.sort(jnp.fft.fftfreq(2**13, d=1/1e4))
detector = GWPhotonCounting.detector.Detector(
    frequencies, '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE1_shot_psd.csv', 
    '/home/ethan.payne/projects/GWPhotonCounting/examples/data/CE1_classical_quanta.csv', 
    gamma=100, random_seed=1632, N_frequency_spaces=10)

# Setting up the likelihood
poisson_likelihood = GWPhotonCounting.distributions.PoissonPhotonLikelihood()
noise_likelihood = GWPhotonCounting.distributions.PhaseQuadraturePhotonLikelihood() 
# Use GWPhotonCounting.distributions.GeometricPhotonLikelihood() if you want a geometric distribution

# Setting up the convolution of the noise and poisson distributions
convolved_likelihood = GWPhotonCounting.distributions.MixturePhotonLikelihood(poisson_likelihood, noise_likelihood)

observed_photons = jnp.zeros(detector.N_total_filters)

# Pick the 30th filter within which to simulate the detection of N photons
observed_photons = observed_photons.at[30].set(1)

# Iterate over the number of photons observed 
# and infer the signal. We then add the distribution to the corner plot 
n_photons = jnp.linspace(0, 5, 6)

cmap = matplotlib.cm.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(n_photons)))

legend_labels = [Line2D([0], [0], color=matplotlib.colors.rgb2hex(colors[i], keep_alpha=True), lw=4, label='Observed photons: {}'.format(n_photons[i])) for i in range(len(n_photons))]

for i, n_photon in enumerate(n_photons):

    fit_lorentzian_n = GWPhotonCounting.inference.PhotonCountingInference(detector, frequencies, convolved_likelihood).run(
        observed_photons*n_photon,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4, f0min=1.5e3)

    fit_lorentzian_n_no_background = GWPhotonCounting.inference.PhotonCountingInference(detector, frequencies, poisson_likelihood, include_background=False).run(
        observed_photons*n_photon,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4, f0min=1.5e3)

    GWPhotonCounting.inference.save_analyses(f'result_n_photons_{n_photon}', fit_pc=fit_lorentzian_n, fit_pc_no_background=fit_lorentzian_n_no_background, outdir='results_higherf0min/',
               n_photons=float(n_photon))
    
    # Generating all the plots
    if n_photon == 0:
        fig_background = corner(fit_lorentzian_n.posterior, plot_datapoints=False, fill_contours=False, plot_density=False, color=matplotlib.colors.rgb2hex(colors[i], keep_alpha=True), levels=[0.9], smooth=0.7, 
            var_names=['A', 'f0', 'gamma', 'phase', 't0'], labels=[r'A [1/$\sqrt{Hz}$]', 'f0 [Hz]', r'$\gamma$ [Hz]', r'$\phi$', r'$t_0$'], 
            hist_kwargs={'density':True}, truths=[None, None, None, None, None], truth_color='k')
        fig_no_background = corner(fit_lorentzian_n_no_background.posterior, plot_datapoints=False, fill_contours=False, plot_density=False, color=matplotlib.colors.rgb2hex(colors[i], keep_alpha=True), levels=[0.9], smooth=0.7, 
            var_names=['A', 'f0', 'gamma', 'phase', 't0'], labels=[r'A [1/$\sqrt{Hz}$]', 'f0 [Hz]', r'$\gamma$ [Hz]', r'$\phi$', r'$t_0$'], 
            hist_kwargs={'density':True}, truths=[None, None, None, None, None], truth_color='k')
    else:
        corner(fit_lorentzian_n.posterior, plot_datapoints=False, fill_contours=False, plot_density=False, color=matplotlib.colors.rgb2hex(colors[i], keep_alpha=True), levels=[0.9], smooth=0.7, 
            var_names=['A', 'f0', 'gamma', 'phase', 't0'], labels=[r'A [1/$\sqrt{Hz}$]', 'f0 [Hz]', r'$\gamma$ [Hz]', r'$\phi$', r'$t_0$'], 
            hist_kwargs={'density':True}, truths=[None, None, None, None, None], truth_color='k', fig=fig_background)
        
        corner(fit_lorentzian_n_no_background.posterior, plot_datapoints=False, fill_contours=False, plot_density=False, color=matplotlib.colors.rgb2hex(colors[i], keep_alpha=True), levels=[0.9], smooth=0.7, 
            var_names=['A', 'f0', 'gamma', 'phase', 't0'], labels=[r'A [1/$\sqrt{Hz}$]', 'f0 [Hz]', r'$\gamma$ [Hz]', r'$\phi$', r'$t_0$'], 
            hist_kwargs={'density':True}, truths=[None, None, None, None, None], truth_color='k', fig=fig_no_background)

fig_background.legend(handles=legend_labels, loc='upper right')
fig_background.savefig('corner_plot_background.pdf', bbox_inches='tight')

fig_no_background.legend(handles=legend_labels, loc='upper right')
fig_no_background.savefig('corner_plot_no_background.pdf', bbox_inches='tight')