import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import matplotlib  as mpl
mpl.rcParams['figure.figsize'] = 13.94, 4.98 # twice the mnras fig size
#                                              for a 2-column figure
mpl.rcParams['font.size'] = 12

def load_accretion(sim):
    filename = os.path.join(sim, 'fluxes.dat')

    def get_key_val(str):
        d = str.strip().split('=')
        return d[0], float(d[1])

    res = {}
    with open(filename) as f:
        for line in f:
            for item in line.split(','):
                k,v = get_key_val(item)
                res[k] = v
    return res


# Ratio of St down-stream of IF to mid-plane St
St_fac = {
    2.5 : 0.122983 / 0.000553029,
    3.0 : 0.0997302 / 0.000134867,
    3.5 : 0.0841049 /  2.83882e-05,
    4.0 : 0.0731963 / 5.2223e-06,
}
# Critical Stokes number (at wind base)
St_c = {
    2.5 : 0.122983,
    3.0 : 0.0997302,
    3.5 : 0.0841049,
    4.0 : 0.0731963, 
}

# Ratio of St up-stream of IF to mid-plane St
St_fac = {
    2.5 : 0.00802037 / 0.000553029,
    3.0 : 0.00650417 / 0.000134867,
    3.5 : 0.00548521 /  2.83882e-05,
    4.0 : 0.00477382 / 5.2223e-06,
}
# Critical Stokes number (at wind base)
St_c = {
    2.5 : 0.00802037,
    3.0 : 0.00650417,
    3.5 : 0.00548521,
    4.0 : 0.00477382,
}

# Mid-plane Mach number
Mach = {
    2.5 : 0.00128609,
    3.0 : 0.000364965,
    3.5 : 8.64946e-05,
    4.0 : 1.74715e-05,
}


Sts = [0, 1e-7, 3.162e-7, 1e-6, 3.162e-6, 1e-5, 3.162e-5, 1e-4, 3.162e-4]
z = 4
if z <= 3:
    Sts = [0, 1e-6, 3.162e-6, 1e-5, 3.162e-5, 1e-4, 3.162e-4, 1e-3, 3.162e-3]

St_fac =  St_fac[z]
St_c = St_c[z]

def St_front(St):
    return St / St_fac


f, axes = plt.subplots(1, 3, sharey=True)

for alpha, ax  in zip([0.05, 0.005, 0.0005], axes):
    for W in [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]:
        
        eps = []
        eps2 = []
        for St in Sts:
            sim = os.path.join('output', 'z_IF{}'.format(z),
                               'alpha{}'.format(alpha),
                               'W{}'.format(W), 'St{}'.format(St))

            data = load_accretion(sim)

            Mdot_d = data['Mdot_dust'] / data['M_dust']
            Mdot_g = data['Mdot_gas'] / data['M_gas']
            

            eps.append(Mdot_d / Mdot_g)
            eps2.append(data['Mdot_dust']  / data['Mdot_dust_advective'])

        l, = ax.semilogx(np.array(Sts) * St_fac, eps, ls='-', marker='^', 
                         label='$W=10^{%d}$ au'%(int(np.log10(W))))
        ax.plot(np.array(Sts) * St_fac, eps2, ls='--', marker='x', 
                c=l.get_color())

    ax.axvline(St_c, ls='--', c='k')

    ax.set_xlabel('St($z_\mathrm{IF}$)')
    if alpha == 0.05:
        ax.set_ylabel('flux efficiency')
    if alpha == 0.0005:
        ax.legend(frameon=False, loc='upper left', fontsize=12)

    ax.text(0.9*St_c, 0.0, r'$St_\mathrm{crit}$',
            verticalalignment='center', horizontalalignment='right')

    ax.set_ylim(top=3.1831)
    
    ax2 = ax.twiny()
    ax2.semilogx(Sts, eps, alpha=0)
    ax2.set_xlabel('$St_\mathrm{mid}$')

    if z > 3:
        ax2.text(3e-4, 3.0, r'$\alpha={}$'.format(alpha), 
                 verticalalignment='center', horizontalalignment='right')
    else:
        ax2.text(3e-3, 3.0, r'$\alpha={}$'.format(alpha), 
                 verticalalignment='center', horizontalalignment='right')


plt.subplots_adjust(top=0.872,
                    bottom=0.12,
                    left=0.05,
                    right=0.989,
                    hspace=0.19,
                    wspace=0.05)
plt.show()
