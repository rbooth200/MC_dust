import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def load_snap(filename):
    data = np.genfromtxt(filename).T

    return dict(zip(['z', 'rho_d', 'rho_g', 'v_d', 'v_g'], data)) 


sim = {
    'sf' :  load_snap('output_diffusion/z_IF3.5/alpha0/St1e-05/results_sf.dat'),
    0 : load_snap('output_diffusion/z_IF3.5/alpha0/St1e-05/results.dat'),
    0.0005 : load_snap('output_diffusion/z_IF3.5/alpha0.0005/St1e-05/results.dat'),
    0.005 : load_snap('output_diffusion/z_IF3.5/alpha0.005/St1e-05/results.dat'),
    0.05 : load_snap('output_diffusion/z_IF3.5/alpha0.05/St1e-05/results.dat'),
}

f, ax = plt.subplots(1,3, figsize=(10,4))

ax[0].semilogy(sim[0]['z'], sim[0]['rho_g'], 'k')
ax[0].set_xlabel('z')
ax[0].set_ylabel('Gas density')

ax[1].semilogy(sim[0]['z'], sim[0]['v_g'], 'k-', label='gas')
ax[1].semilogy(sim[0]['z'], sim[0]['v_d'], 'k--', label='dust')
ax[1].semilogy(sim['sf']['z'], sim['sf']['v_d'], 'k:', label='terminal velocity')

ax[1].set_xlabel('z')
ax[1].set_ylabel('Velocity')
ax[1].legend(loc='lower right')

ax[2].plot(sim[0]['z'], sim[0]['rho_d']/sim[0]['rho_g'], 'k-')
ax[2].plot(sim[0]['z'], sim['sf']['rho_d']/sim[0]['rho_g'], 'k--')
ax[2].set_xlabel('z')
ax[2].set_ylabel('Dust-to-gas ratio')

plt.tight_layout()

ax1_inset = inset_axes(ax[1], width='40%',height='40%', loc='upper left', borderpad=3)
ax1_inset.semilogy(sim[0]['z'], sim[0]['v_g'], 'k-', label='gas')
ax1_inset.semilogy(sim[0]['z'], sim[0]['v_d'], 'k--', label='dust')
ax1_inset.semilogy(sim['sf']['z'], sim['sf']['v_d'], 'k:', label='terminal velocity')

ax1_inset.set_xlim(3.49, 3.51)
ax1_inset.set_ylim(ymin=0.01)



ax2_inset = inset_axes(ax[2], width='40%',height='70%', loc='upper left', borderpad=2)
ax2_inset.plot(sim[0]['z'], sim[0]['rho_d']/sim[0]['rho_g'], 'k-')
ax2_inset.plot(sim[0]['z'], sim['sf']['rho_d']/sim[0]['rho_g'], 'k--')

ax2_inset.set_xlim(3.49, 3.51)


plt.show()
