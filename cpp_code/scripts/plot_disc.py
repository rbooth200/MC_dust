import numpy as np
import matplotlib.pyplot as plt

def load_disc(filename):
    return np.genfromtxt(filename) 

def plot_disc(disc, ax, **kwargs):
    for i in range(1, disc.shape[1]):
        ax.plot(disc[:,0], disc[:,i], **kwargs)

    ax.set_xlabel('z')
    ax.set_ylabel('density')

if __name__ == "__main__":
    import sys

    f, ax = plt.subplots(1,1)

    for f in sys.argv[1:]:
        h = load_disc(f)
        plot_disc(h, ax, label='f')

    ax.set_yscale('log')
    plt.show()
        