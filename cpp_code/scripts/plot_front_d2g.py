import numpy as np
import matplotlib.pyplot as plt

def load_front(filename):
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

def combine_front_model(files):
    hist = {}
    hist['counts'] = 0
    hist['mean_v'] = 0
    var = 0 ; n = 0
    for f in files:
        data = np.genfromtxt(f) 
        hist['z'] = data[:,0]
        hist['counts'] += data[:,1]
        hist['mean_v'] += data[:,1]*data[:,2]
        var += (data[:, 3]**2 + data[:,2]**2)*data[:,1]
        n += 1
    hist['mean_v'] /= np.maximum(hist['counts'], 1)
    var = var/np.maximum(hist['counts'], 1) - hist['mean_v']**2

    hist['freq'] = hist['counts'] / (np.diff(hist['z']).mean() * hist['counts'].sum())
    hist['std_v'] = np.sqrt(np.maximum(var,0))
    hist['n'] = n

    disc = {}
    disc['z'] = data[:,0]
    disc['rho'] = data[:, 4]
    disc['v'] = data[:, 5]

    return  { 'gas' : disc, 'hist' : hist}

def rebin_data(hist, Nsub=100):
    z, counts, v, v2 = [], [], [],[]
     
    v_i = hist['mean_v']*hist['counts']
    v2_i = (hist['std_v']**2 + hist['mean_v']**2)*hist['counts']

    start = 0
    end = Nsub
    while start < len(hist['z']):
        z.append(hist['z'][start:end].mean())
        counts.append(hist['counts'][start:end].sum())
        v.append(v_i[start:end].sum())
        v2.append(v2_i[start:end].sum())

        start += Nsub 
        end += Nsub 
    
    counts = np.array(counts)
    mean_v = np.array(v) / np.maximum(counts,1)
    var = np.array(v2) / np.maximum(counts,1) - mean_v**2

    res = {
        'z' : np.array(z),
        'counts' : counts,
        'freq' :  counts / (np.diff(z).mean()*counts.sum()),
        'mean_v' : mean_v,
        'std_v' : np.sqrt(np.maximum(var, 0))
    }
    return res


def rebin_gas(gas, Nsub=100):
    z, rho, v =[], [], []
     
    start = 0
    end = Nsub
    while start < len(gas['z']):
        z.append(gas['z'][start:end].mean())
        rho.append(gas['rho'][start:end].mean())
        v.append((gas['rho']*gas['v'])[start:end].mean()/rho[-1])

        start += Nsub 
        end += Nsub 
    
    res = {
        'z' : np.array(z),
        'rho' : np.array(rho),
        'v' : np.array(v),
    }
    return res




def plot_front(front, ax0, ax1, label=None):

    gas = front['gas']

    hist = front['hist']
    hist2 = rebin_data(hist)
    gas2 = rebin_gas(gas)

    norm = gas['rho'].sum() * np.diff(gas['z']).mean()

    xerr = 0.5*np.diff(hist['z']).mean()      
    xerr2 = 0.5*np.diff(hist2['z']).mean()      

    ax0.semilogy(gas['z'], gas['rho'], 'k-', label='gas', zorder=1)
    ax0.errorbar(hist['z'], hist['freq']*norm,
                 xerr=xerr, marker='+', ls='',zorder=0)
    ax0.errorbar(hist2['z'], hist2['freq']*norm,
                 xerr=xerr2, marker='+', ls='',zorder=2)

    ax0.set_xlabel('z')
    ax0.set_ylabel('density')

    err = hist['std_v'] / np.maximum(hist['counts']-1, 1)**0.5
    err2 = hist2['std_v'] / np.maximum(hist2['counts']-1, 1)**0.5
  
    ax1.errorbar(hist['z'], hist['freq']*norm/gas['rho'],
                 xerr=xerr, marker='+', ls='',zorder=0)
    ax1.errorbar(hist2['z'], hist2['freq']*norm/gas2['rho'],
                 xerr=xerr2, marker='+', ls='',zorder=2)

    ax1.set_xlabel('z')
    ax1.set_ylabel('dust-to-gas ratio')

if __name__ == "__main__":
    import sys
    
    front = combine_front_model(sys.argv[1:])
    print("Showing results of {} models".format(front['hist']['n']))

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4,6))

    plot_front(front, ax[0], ax[1])
    ax[0].legend()
   
    fig.set_tight_layout(True)

    plt.show()

