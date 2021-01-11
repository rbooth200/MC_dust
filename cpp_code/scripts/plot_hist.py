import numpy as np
import matplotlib.pyplot as plt

def load_hist(filename):
    data = np.genfromtxt(filename) 

    hist = {}
    hist['z'] = data[:,0]
    hist['counts'] = data[:, 1]
    hist['freq'] = data[:, 1] / (np.diff(hist['z']).mean() * data[:,1].sum())
    hist['mean_v'] = data[:, 2]
    hist['std_v'] = data[:, 3]

    return hist

def plot_hist(hist, ax, **kwargs):
    ax.plot(hist['z'], hist['freq'], **kwargs)

    ax.set_xlabel('z')
    ax.set_ylabel('density')

if __name__ == "__main__":
    import sys

    f, ax = plt.subplots(1,1)

    for f in sys.argv[1:]:
        h = load_hist(f)
        plot_hist(h, ax, marker='+', ls='none', label='f')

    z = np.linspace(-2,2, 10000)
    H = 1; St = 0.05
    Hp = H / np.sqrt((1 + St/1e-2) * (1 + St/(1 + St)))
    ax.plot(z, np.exp(-0.5*(z/Hp)**2) / (np.sqrt(2*np.pi)*Hp), c='k')

    ax.set_yscale('log')
    plt.show()
        