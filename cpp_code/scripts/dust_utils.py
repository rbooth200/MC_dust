import numpy as np
import os

def load_snap(Sim, snap):
    with open(os.path.join(Sim, 'snap', 'snap_{}.dat'.format(snap)), 'rb') as f:
        f.readline()
        data = f.read()
    time = np.fromstring(data[:8], 'f8')[0]
    N = np.fromstring(data[8:12], 'i4')[0]

    dtype = np.dtype([('z', 'f8'), ('v', 'f8'), ('vt', 'f8'),('level', 'i4')])
    data = np.fromstring(data[12:], dtype)
       
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
    disc['vd'] = data[:,6]

    return  { 'gas' : disc, 'hist' : hist}


def combine_histograms(sim, start, end):
    hist = {}
    hist['counts'] = 0
    hist['mean_v'] = 0
    var = 0 ; n = 0
    for i in range(start, end):
        data = load_hist(sim, i)
        h = data['hist']
        hist['counts'] += h['counts']
        hist['mean_v'] += h['counts']*h['mean_v'] 

        var += (h['mean_v']**2 + h['std_v']**2)*h['counts']
        n += 1
    
    hist['z'] = h['z']
    hist['mean_v'] /= np.maximum(hist['counts'], 1)
    var = var/np.maximum(hist['counts'], 1) - hist['mean_v']**2

    dz = np.diff(hist['z']).mean()
    hist['freq'] = hist['counts'] / (dz * hist['counts'].sum())
    hist['std_v'] = np.sqrt(np.maximum(var,0))
    hist['n'] = n

    return { 'gas' : data['gas'], 'hist' : hist }


def rebin_data(data, Nsub=100):
    z, counts, v, v2, = [], [], [],[]
    rho, vg, vd =  [],[],[]

    hist = data['hist']
    gas = data['gas']
    

    v_i = hist['mean_v']*hist['counts']
    v2_i = (hist['std_v']**2 + hist['mean_v']**2)*hist['counts']

    start = 0
    end = Nsub
    while start < len(hist['z']):
        z.append(hist['z'][start:end].mean())
        counts.append(hist['counts'][start:end].sum())
        v.append(v_i[start:end].sum())
        v2.append(v2_i[start:end].sum())

        rho.append(gas['rho'][start:end].mean())
        vg.append((gas['rho']*gas['v'])[start:end].mean()/rho[-1])
        vd.append(gas['vd'][start:end].mean())

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

    gas = { 'z' : np.array(z), 'rho' : np.array(rho),
            'v' : np.array(vg), 'vd' : np.array(vd) }

    return { 'hist' : res, 'gas' : gas }

