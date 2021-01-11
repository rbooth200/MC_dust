import numpy as np
import matplotlib.pyplot as plt
import matplotlib  as mpl
mpl.rcParams['figure.figsize'] = 13.94, 4.98 # twice the mnras fig size
#                                              for a 2-column figure
mpl.rcParams['font.size'] = 12

import os
from scipy.interpolate import PchipInterpolator
from scipy.optimize import bisect

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

Sts = [0, 1e-7, 3.162e-7, 1e-6, 3.162e-6, 1e-5, 
       3.162e-5, 1e-4, 3.162e-4, 1e-3, 3.162e-3]


def comput_St_crit(z, alpha, W=1e-5):
    sim_base =  os.path.join('output', 'z_IF{}'.format(z),
                        'alpha{}'.format(alpha),
                        'W{}'.format(W))

    return _compute_St_crit(sim_base) * St_fac[z]

def _compute_St_crit(sim_base):

    eps = []
    mySt = []
    for St in Sts[1:]:
        try:
            sim = os.path.join(sim_base, 'St{}'.format(St))
            data = load_accretion(sim)
        except FileNotFoundError:
            continue 
        Mdot_d = data['Mdot_dust'] / data['M_dust']
        Mdot_g = data['Mdot_gas'] / data['M_gas']
            

        mySt.append(St)
        eps.append(Mdot_d / Mdot_g)
    
    try:
        f = PchipInterpolator(np.log(mySt), 
                              np.array(eps) - 0.5, extrapolate=True)
        St_crit = bisect(f, np.log(mySt[0]), np.log(3*mySt[-1]))
    except (ValueError, IndexError):
        print(sim_base)
        print(eps)
        St_crit = np.nan
    return np.exp(St_crit)


W = 1e-5
f, axes = plt.subplots(1, 2)

# Models with fixed R
zs =  [2.5, 3, 3.5, 4]
for alpha  in [0.05, 0.005, 0.0005]:
    St_crit_an = []
    St_crit_num = []
    for z in zs:
        St_crit_num.append(comput_St_crit(z, alpha, W))
        St_crit_an.append(St_c[float(z)])

    if alpha == 0.005:
        axes[0].plot(zs, St_crit_an, 'k', label=r'$St_{\rm crit}$')
    if alpha == 0.0005:
        axes[0].plot(zs, 0.5/np.array(zs), 'k--', label=r'$St_{\rm max}$')
    axes[0].semilogy(zs, St_crit_num, label=r'$\alpha={}$'.format(alpha))

axes[0].text(2.5, 2.5e-3, r'$R=10\,\mathrm{au}$',
             verticalalignment='bottom', horizontalalignment='left')
axes[0].legend(ncol=3, loc='upper center', frameon=False)
axes[0].set_xlabel(r'$z_{\rm IF}/H$')
axes[0].set_ylabel(r'$St$')
axes[0].set_ylim(2e-3, 0.7)

# Models with fixed ionization rate
St_fac = {
    4   : 0.00561384 / 9.25048e-06,
    10  : 0.00534636 / 2.15935e-05,
    20  : 0.00516733 / 4.95292e-05,
    40  : 0.00507746 / 0.00011264,
    100 : 0.00506517 / 0.000247029,
}

St_c = {
    4   : 0.00561384,
    10  : 0.00534636, 
    20  : 0.00516733,
    40  : 0.00507746,
    100 : 0.00506517,
}
zIF = {
    4   : 3.82583,
    10  : 3.58337,
    20  : 3.3288,
    40  : 3.05297,
    100 : 2.75797,
}

rho_0 = {
    4   : 0.131409,
    10  : 0.139747,
    20  : 0.151359,
    40  : 0.163872,
    100 : 0.181541,
}

Radii = {
    4   : 4.6416,
    10  : 10,
    20  : 21.544,
    40  : 46.416,
    100 : 100
}

def comput_St_crit(R, alpha):
    sim_base =  os.path.join('full_disc_sim/', 
                        'alpha{}'.format(alpha), 'ion_flux1e+42',
                        'R{}'.format(Radii[R]))

    return _compute_St_crit(sim_base)  * St_fac[R]

Rs = [4, 10, 20, 40, 100]
for alpha  in [0.05, 0.005, 0.0005]:
    St_crit_an = []
    St_crit_num = []
    rs = []
    zs = []
    for R in Rs:
        St_crit_num.append(comput_St_crit(R, alpha))
        St_crit_an.append(St_c[R])
        rs.append(Radii[R])
        zs.append(zIF[R])

    if alpha == 0.005:
        axes[1].plot(rs, St_crit_an, 'k', label=r'$St_{\rm crit}$')
    if alpha == 0.0005:
        axes[1].plot(rs, 0.5/np.array(zs), 'k--', label=r'$St_{\rm max}$')
    axes[1].semilogy(rs, St_crit_num, label=r'$\alpha={}$'.format(alpha))


axes[1].set_xlabel(r'$\mathrm{Radius}\,[\mathrm{au}]$')
axes[1].set_ylabel(r'$St$')
axes[1].set_xlim(3, 120)
axes[1].set_ylim(2e-3, 0.7)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].legend(ncol=3, loc='upper center', frameon=False)


plt.tight_layout()

plt.show()

