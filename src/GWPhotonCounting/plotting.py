import jax.numpy as jnp

# plotting imports
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from copy import copy
params = {
   'axes.labelsize': 9,
   'font.size': 9,
   'legend.fontsize': 9,
   'xtick.labelsize': 9,
   'ytick.labelsize': 9,
   'axes.titlesize':9,
   'text.usetex': True,
   'font.family':'serif',
   'font.serif':'Computer Modern'
   }
matplotlib.rcParams.update(params)
matplotlib.rcParams["font.serif"] = "Computer Modern Roman"
matplotlib.rcParams["font.family"] = "Serif"
matplotlib.rcParams['text.latex.preamble'] = r'\renewcommand{\mathdefault}[1][]{}'

from matplotlib.markers import MarkerStyle
import matplotlib as mpl


def generate_count_plot(data, detector):

    # TODO customize labels
    # TODO allow for discrete or continuous colorbar
    # TODO add second plot for the strain signal with the color-coded filters overlaid on top (maybe a second function)

    points = jnp.array([(float(f0), float(t0)) for f0 in detector.f0_values for t0 in detector.t0_values])

    color_values = [[0,0] for _ in range(int(detector.N_total_filters/2))]

    for i in range(detector.N_total_filters):
        
        label = detector.filter_labels[i]
        
        point_idx = int(jnp.argwhere(jnp.sum(jnp.array(points) - jnp.array(label[0:2]),axis=1) == 0)[0][0])
        
        if label[2] > 0:
            color_values[point_idx][1] = (data[i]/jnp.max(data))
            
        else:
            color_values[point_idx][0] = (data[i]/jnp.max(data))
            
    color_values = jnp.array(color_values)

    cmap = plt.cm.Purples
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = jnp.linspace(0, jnp.max(data), 100)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(3.375, 4))

    ax = plt.gca()
    plt.scatter(points.T[0], points.T[1], c=cmap(color_values.T[0]), edgecolor="k", marker=MarkerStyle("o", fillstyle="right"), s=70)
    plt.scatter(points.T[0], points.T[1], c=cmap(color_values.T[1]), edgecolor="k", marker=MarkerStyle("o", fillstyle="left"), s=70)

    ax2 = fig.add_axes([0.125, 0.9, 0.7775, 0.03])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, label=r'$\bar{n}_{k}$ [quanta]', norm=norm,
                                boundaries=bounds, format='%.2f', orientation='horizontal')


    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')

    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_ylabel(r'Time offset [s]')

    return fig

