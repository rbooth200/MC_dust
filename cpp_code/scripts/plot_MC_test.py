import os
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib  as mpl
mpl.rcParams['figure.figsize'] = 13.94, 4.98 # twice the mnras fig size
#                                              for a 2-column figure
mpl.rcParams['font.size'] = 12

def load_hist(filename):
    data = np.genfromtxt(filename) 

    hist = {}
    hist['z'] = data[:,0]
    hist['counts'] = data[:, 1]
    hist['freq'] = data[:, 1] / (np.diff(hist['z']).mean() * data[:,1].sum())
    hist['mean_v'] = data[:, 2]
    hist['std_v'] = data[:, 3]

    return hist


def plot_test(ax, St, strat=False, legend=False, **kw):
    if strat:
        DIR = 'test_results/strat_St{}'.format(St)
    else:
        DIR = 'test_results/simple_St{}'.format(St)

    sims = ['thomson', 'ormel', 'laibe']
    labels = ['Thomson (1984)', 'Ormel & Liu (2018)', 'Laibe et al. (2020)']
    for sim, label in zip(sims, labels):
        try:
            hist = load_hist(os.path.join(DIR, sim + '.dat'))
        except OSError:
            continue

        ax.semilogy(hist['z'], hist['freq'], '+', label=label, **kw)
    
    if legend:
        ax.legend()
    
    if strat:
        ax.text(hist['z'].min()*0.95, 1.0, '$St={}$\n'.format(St)+r'Variable $\alpha$',
                verticalalignment='center', horizontalalignment='left')
    else:
        ax.text(hist['z'].min()*0.95, 1.0, '$St={}$\n'.format(St)+r'Constant $\alpha$',
                verticalalignment='center', horizontalalignment='left')

    # Add the analytic solution:
    #   Youdin & Lithwick (2007)
    z = hist['z']
    Hp = 1 / np.sqrt((1 + St/1e-2) * (1 + St/(1 + St)))
    rho_z = np.exp(-0.5*(z/Hp)**2) / (np.sqrt(2*np.pi)*Hp)
    ax.plot(z, rho_z, c='k', zorder=10)


f, axes = plt.subplots(1, 3, figsize=(10,4), sharey=True)

strat = [False, True, True]
St = [0, 0, 0.05]

for St_i, strat_i, ax in zip(St, strat, axes):
    plot_test(ax, St_i, strat_i)
    ax.set_xlabel('$z/H$')
axes[0].set_ylabel(r'$\rho$')
axes[0].set_ylim(1e-4, 2) 
axes[2].legend(frameon=True)

plt.subplots_adjust(top=0.975, bottom=0.115,
                    left=0.08, right=0.98,
                    hspace=0.2, wspace=0.05)

plt.show()
