import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter, LogLocator
import matplotlib
matplotlib.rcParams['figure.figsize'] = 6.64, 4.98 # twice the mnras fig size
#                                                    for a 1-column figure
matplotlib.rcParams['font.size'] = 12


G      = 6.67e-7
Ms     = 1.989e33
mH     = 1.67e-24
alpha2 = 2.6e-13
au     = 1.5e13
yr     = 3600*24*365.25

z_IF = 4.
rho_grain = 1.
cs_w   = 12.86e5
M_w    = 0.5

def Sigmadot_EUV(Rau, Phi, Rg=5.4):
    R = Rau*au
    return 0.2 * mH * np.sqrt(3*Phi/(4*np.pi*alpha2*R**3)) * \
        M_w * cs_w * np.minimum(1, Rg/Rau)

def Sigmadot_Xray(Rau, Lx):
    a=-0.5885
    b=4.3130
    c=-12.1214
    d=16.3587
    e=-11.4721
    f=5.7248
    g=-2.8562

    def f0(x):
        return g + x*(f + x*(e + x*(d + x*(c + x*(b + x*a)))))
    def fp(x):
        return f + x*(2*e + x*(3*d + x*(4*c + x*(5*b + x*6*a))))

    AL=-2.7326
    BL=3.3307
    CL=-2.9868e-3
    DL=-7.2580

    def M(Lx):
        return 10**(AL * np.exp((np.log(np.log10(Lx)) - BL)**2 / CL) + DL)
    
    lgR = np.log10(Rau)
    Sigdot = M(Lx) * 10**f0(lgR) * fp(lgR) / (2*np.pi * R**2)

    #return M(Lx) * 10**f0(lgR)
    
    return Sigdot * Ms / (au*au*yr)


def s_crit(Rau, Sigmadot):
    Omega = 2*np.pi/yr * Rau**-1.5
    return np.sqrt(8/np.pi) * (Sigmadot / (rho_grain * Omega)) * (1 / z_IF)


f = plt.figure()
ax = plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')


R = np.geomspace(0.7, 100, 100)
SigmaDot = np.logspace(-16, -10, 200)



RR, SS = np.meshgrid(R, SigmaDot)
plt.pcolormesh(R, SigmaDot, 1e4*s_crit(*np.meshgrid(R, SigmaDot)),
             norm=LogNorm())  
             #levels=10.**np.arange(-6,5,1),
             #locator=LogLocator())
plt.colorbar(label='$s_\mathrm{crit}\;[\mathrm{micron}]$')
CS = plt.contour(R, SigmaDot, 1e4*s_crit(*np.meshgrid(R, SigmaDot)),
                 levels=10.**np.arange(-5,4,1), colors='w', linewidths=0.5)
def fmt(val):
    return '$10^{' +'{}'.format(np.log10(val))[:-2] + '}$'

labels = plt.clabel(CS, fmt=fmt, inline=1)#, use_clabeltext=True)

plt.loglog(R, Sigmadot_EUV(R, 1e42), c='k', label='EUV')
plt.loglog(R, Sigmadot_Xray(R, 2e30), c='k',label='Xray')
plt.text(1, 1.5e-11, "X-ray")
plt.text(40, 2.5e-15, "EUV")


plt.xlabel('$R\;[\mathrm{au}]$')
plt.ylabel('$\dot{\Sigma}\;[\mathrm{g\,cm^{-2}\,s^{-1}}]$')

plt.tight_layout()


for ax in f.get_axes():
    ax.set_rasterized(True)

plt.subplots_adjust(top=0.97, bottom=0.11,
                    left=0.14, right=0.97,
                    hspace=0.2, wspace=0.2)

plt.show()
