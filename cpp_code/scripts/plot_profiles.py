import numpy as np
import matplotlib.pyplot as plt
import matplotlib  as mpl
mpl.rcParams['figure.figsize'] = 13.94, 8.364  # twice the mnras fig size
#                                              for a 2-column figure
mpl.rcParams['font.size'] = 12

import os
from dust_utils import combine_histograms, rebin_data


f, ax = plt.subplots(3,3, sharex='all', sharey='row')
f.subplots_adjust(top=0.984, bottom=0.068,
                    left=0.06, right=0.985,
                    hspace=0.055,wspace=0.030)
sims = [
    'output/z_IF3.5/alpha0.0005/W1e-05/St0',
     'output/z_IF3.5/alpha0.0005/W1e-05/St1e-05',
    'output/z_IF3.5/alpha0.05/W1e-05/St1e-05',
]
labels = [
    r'$\alpha=5\times10^{-4}$' + '\n' + r'$St_\mathrm{mid}=0$',
    r'$\alpha=5\times10^{-4}$' + '\n' + r'$St_\mathrm{mid}=10^{-5}$',
    r'$\alpha=5\times10^{-2}$' + '\n' + r'$St_\mathrm{mid}=10^{-5}$',
]
for i, sim in enumerate(sims):
    data = rebin_data(combine_histograms(sim, 50, 101), 25)
    gas = data['gas']
    dust = data['hist']

    norm_g = np.sum(gas['rho'] * np.diff(gas['z']).mean())

    ax[0][i].semilogy(dust['z'], dust['freq'], marker='+', ls='')
    ax[0][i].semilogy(gas['z'], gas['rho']/norm_g, c='k', ls='-')   

    rho_d = gas['rho']*gas['v']/gas['vd']
    rho_d /= np.sum(rho_d * np.diff(gas['z']).mean())

    mask = gas['z'] < 5
    ax[1][i].plot(gas['z'][mask], (rho_d/(gas['rho']/norm_g))[mask],
                  c='k', ls='--')
    ax[1][i].plot(dust['z'], dust['freq']/(gas['rho']/norm_g),
                  marker='+',  ls='')

    
    ax[2][i].semilogy(dust['z'], dust['mean_v'], marker='+', ls='')
    ax[2][i].semilogy(gas['z'], gas['v'], c='k', ls='-',
                      label='gas velocity')
    ax[2][i].semilogy(gas['z'], gas['vd'], c='k', ls='--', 
                      label='terminal velocity')

    if i == 0:
        ax[0][i].set_ylabel('Density')
        ax[1][i].set_ylabel('Dust-to-gas ratio')
        ax[2][i].set_ylabel('Velocity [$c_\mathrm{s,mid}$]')
        ax[2][i].legend(loc='best')

    ax[2][i].set_xlabel('$z/H$')
    ax[1][i].set_ylim(0.78, 1.22)

    ax[0][i].text(5.0, 0.8, labels[i], 
                  verticalalignment='top', horizontalalignment='right')
                  

plt.show()
