import numpy as np
#import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def load_snap(Sim, snap):
    with open(os.path.join(Sim, 'snap', 'snap_{}.dat'.format(snap)), 'rb') as f:
        f.readline()
        data = f.read()
    time = np.frombuffer(data[:8], 'f8')[0]
    N = np.frombuffer(data[8:12], 'i4')[0]

    dtype = np.dtype([('z', 'f8'), ('v', 'f8'), ('vt', 'f8'),('level', 'i4')])
    data = np.frombuffer(data[12:], dtype)
       
    return { 't' : time, 'N': N, 
             'z' : data['z'], 'v' : data['v'], 'vt' : data['vt'], 
             'level' : data['level'] }
    
def load_hist(sim, number):
    filename = os.path.join(sim, 'binned', 'hist_{}.dat'.format(number))
    data = np.genfromtxt(filename) 
    
    hist = {}
    hist['z'] = data[:,0]
    hist['counts'] = data[:, 1]
    hist['freq'] = data[:, 1] / (np.diff(hist['z']).mean() * data[:,1].sum())
    hist['mean_v'] = data[:, 2]
    hist['std_v'] = data[:, 3]

    disc = {}
    disc['z'] = data[:,0]
    disc['rho'] = data[:, 4]
    disc['v'] = data[:, 5]

    return  { 'gas' : disc, 'hist' : hist}

def get_num_snaps(Sim):
    snaps = os.listdir(os.path.join(Sim, 'snap'))
    nums = list(map(lambda s: int(s[5:-4]), snaps))
    return max(nums)
                       

def get_disc_properties(Sim):
    data = \
        np.genfromtxt(os.path.join(Sim, 'binned', 'hist_10.dat')).T
 
    return { 'z' : data[0], 'rho' : data[4], 
            'v_gas' : data[5], 'v_dust' : data[6]}

def get_num_density(sim, snap=-1):
    if snap == -1:
        files = os.listdir(os.path.join(sim, 'binned'))
        snap = max(list(map(lambda x:int(x[5:-4]), files)))
        
    data = load_hist(sim, snap)['hist']
    return data['z'], data['freq']
    
def fit_flux(t, N, Ndot, N0):

    def Nt(t, eps, t0=0):
        dt = t - t0
        return (N0 + Ndot*t0) * np.exp(-eps*dt) - (Ndot/eps) * np.expm1(-eps*dt)

    p, cov = curve_fit(Nt, t, N, p0=(1e-3,0), sigma = N**0.5)
    return p[0], lambda t: Nt(t,p[0], p[1])

def get_label(sim):
    St = sim[2+sim.find('St'):]
    return r'{:.3g}'.format(float(St)) #*  0.0918335 / 3.29663e-05)

if __name__ == "__main__":
    import sys 

    Ndot = 4.
    N0 = 16.0
    dt_snap = 1000 * 1.0

    sims = sorted(sys.argv[1:], key=lambda x: float(get_label(x)))
    
    #f, ax = plt.subplots(2, sharex=True)
    for sim in sims:
        print("Loading data for simulation {}...".format(sim))
        num_snaps = get_num_snaps(sim)

        snaps = range(1, num_snaps+1,1)
    
        times, N = [], []
        for s in snaps:
            try:
                data = load_snap(sim, s)
            except FileNotFoundError:
                continue
            times.append(data['t'])
            N.append(data['N'])

        times = np.array(times)
        N = np.array(N)

        print ('Fitting data, making plots')
        Mdot_fit, Nfit = fit_flux(times, N, Ndot, N0)

        disc = get_disc_properties(sim)
        Mdot_gas = disc['v_gas'][0] * disc['rho'][0]
        M_gas = disc['rho'].sum() * np.mean(np.diff(disc['z']))

    
        #l, = ax[0].plot(times, N, label=get_label(sim))
        #ax[0].loglog(times, N_tot, ls='--', c=l.get_color())
        #ax[0].loglog(times, Nfit(times), ls='--', c=l.get_color())
        #ax[0].set_ylabel('$N_\mathrm{part}$')

        Ndot_diff = Ndot - np.diff(N) / np.diff(times)
        #ax[1].plot(times[1:], (Ndot_diff / N[1:])* (M_gas/Mdot_gas), 
        #           label=get_label(sim))
        #ax[1].axhline(Mdot_fit * (M_gas/Mdot_gas), ls='--', c=l.get_color())
        #ax[1].set_ylabel('Flux efficiency')

        #ax[-1].set_xlabel('time')

        print('Saving fluxes...')
        n = get_num_density(sim)[1]
        with open(sim + '/fluxes.dat', 'w') as f:
            print('Mdot_gas={}, M_gas={}'.format(Mdot_gas, M_gas), file=f)
            print('Mdot_dust={}, M_dust={}'.format(Mdot_fit, 1.0), file=f)
            print('Mdot_dust_advective={}'.format(disc['v_gas'][0]*n[0]), file=f)

    #ax[1].axhline(1.0, c='k', ls='--')
    #ax[0].legend(ncol=3, frameon=False)
    #plt.tight_layout()
    #plt.show()
