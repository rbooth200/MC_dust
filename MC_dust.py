import numpy as np
import matplotlib.pyplot as plt


class DiscModel:
    """Gaussian disc model with stratified turbulence:
 
       \rho(z) = \rho_0 * np.exp(- 0.5*(z/H)**2)   
       v_z(z)  = 0
       \alpha(z) = 0.5*(\alpha_0 + \alpha_1) 
           + 0.5*(\alpha_1 - \alpha_0)*np.tanh( (z - z_t) / W)

    The turbulent velocity is assumed to have the form:
        v_t(z) = \sqrt{alpha(z)} * c_s,
    where c_s = H * Omega.
    """
    def __init__(self, Omega=1.0, H=1.0, rho_0=1.0,
                 alpha_0=0.001, alpha_1=0.1, z_t=1.0, W=1e-2, t_eddy=1.0):

        self.rho_0 = rho_0
        self.H = H
        self.Omega = Omega

        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.z_t = z_t
        self.W = W

        self.t_eddy = t_eddy

    def rho(self, z):
        """Density"""    
        return self.rho_0 * np.exp( - 0.5*(z/self.H)**2)
    
    def dlnrho_dz(self, z):
        """Log-gradient of density"""
        return - z / self.H**2

    def v_z(self, z):
        """Vertical velocity"""
        return 0

    def a_grav(self, z):
        """Gravitational acceleration"""
        return - self.Omega**2 * z

    def _alpha(self, z):
        """Turbulent alpha parameter"""
        a0, a1 = self.alpha_0, self.alpha_1
        z_t, W = self.z_t, self.W
        return 0.5*(a0 + a1 + (a1 - a0) * np.tanh((np.abs(z) - z_t)/W))
    
    def _dalpha_dz(self, z):
        """Gradient of alpha"""
        a0, a1 = self.alpha_0, self.alpha_1
        z_t, W = self.z_t, self.W
        return np.sign(z) * 0.5 * (a1 - a0) / (W * np.cosh((np.abs(z)-z_t)/W)**2)
        
    def v_turb_sqd(self, z):
        """Turbulent velocity (squared)"""
        return self._alpha(z) * self.H**2 * self.Omega
    
    def dv_turb_sqd_dz(self, z):
        """derivative of turbulent velocity (squared)"""
        return self._dalpha_dz(z) * self.H**2 * self.Omega


class MonteCarloModel:
    def __init__(self, disc, St, seed=None):
        self.disc = disc
        self.St = St
        self._sample = False

        if seed:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()            

    def setup_sampling(self, dt_sample):
        """Setup sampling the MC distribution"""
        self._dt_sample = dt_sample
        self._sample = True
        
        self._reset_samples()

    def _reset_samples(self):
        self._samples = {
            'z' : [],
            'vd' : [],
            'vt' : [],
        }
                
    def _record_sample(self, particles):

        z, vd, vt = particles
        self._samples['z'] = np.append(self._samples['z'], z)
        self._samples['vd'] = np.append(self._samples['vd'], vd)
        self._samples['vt'] = np.append(self._samples['vt'], vt)
                    
    def get_samples(self):
        """Get sampled density """
        return self._samples

    def __call__(self, particles, Nstep, dt):

        t_sample=0
        for i in range(Nstep):
            particles = self._take_step(particles, dt)

            t_sample += dt

            if self._sample:
                if t_sample > self._dt_sample:
                    self._record_sample(particles)
                    t_sample -= self._dt_sample
                    
        return particles

      
class Ciesla10Model(MonteCarloModel):
    """Update Particles using the model of Ciesla (2010)"""
    def __init__(self, disc, St):
        MonteCarloModel.__init__(self, disc, St)

    def _take_step(self, particles, dt):
        z, vd, vt = particles

        disc = self.disc
        
        ts = self.St * disc.rho_0 / (disc.Omega * disc.rho(z))
        t_e = disc.t_eddy
        D = disc.v_turb_sqd(z)*t_e

        Un = disc.v_z(z) + D*disc.dlnrho_dz(z)
        Ufp = disc.dv_turb_sqd_dz(z)*t_e 
        Ug = disc.a_grav(z) * ts

        zp = z + 0.5*dt*Ufp
        D = disc.v_turb_sqd(zp)*t_e

        vt = np.sqrt(2*D/dt) * self._rng.standard_normal(size=len(z))
        vd = Un + Ufp + Ug + vt
        z += vd * dt

        return z, vd, vt  
    
class Ormel18Model(MonteCarloModel):
    """Update Particles using the model of Ormel & Liu (2018)

    Either equations 4 (SCA=False) or 10 (SCA=True) are used.
    """
    def __init__(self, disc, St, SCA=False):
        MonteCarloModel.__init__(self, disc, St)
        self.SCA = SCA

    def _take_step(self, particles, dt):
        z, vd, vt = particles

        disc = self.disc
        
        t_e = disc.t_eddy
        D = disc.v_turb_sqd(z)*t_e

        Un = disc.v_z(z)
        dD = disc.dv_turb_sqd_dz(z)*t_e 
        v_hs = 0.5*dD + D*disc.dlnrho_dz(z)

        ts = self.St * disc.rho_0 / (disc.Omega * disc.rho(z))

        z += vd*dt
        if self.SCA:
            vt = np.sqrt(2*D/dt) * self._rng.standard_normal(size=len(z))
            vd = disc.a_grav(z) * ts + Un + v_hs + 0.5*dD/(1+ts/t_e) + vt
        else:
            vg = Un + v_hs + vt*np.sqrt(D/t_e)
            vd = ((vd + disc.a_grav(z)*dt)*ts + vg*dt) / (ts + dt)

            # Using integrated update from Laibe+ (2020)
            x = dt / disc.t_eddy
            dvt = np.sqrt(-np.expm1(-2*x))
            vt = np.exp(-x)*vt + dvt*self._rng.standard_normal(size=len(z))

        return z, vd, vt
    
class Laibe20Model(MonteCarloModel):
    """Update Particles using equation 84 of Laibe et al. (2020)"""
    def __init__(self, disc, St):
        MonteCarloModel.__init__(self, disc, St)

    def _drift(self, dt, particles):
        z, vd, vt = particles
        disc = self.disc

        # Step 1: drift the particles
        z += vd*dt

        # Step 2: Update the turbulent (gas) velocity
        ve2 = disc.v_turb_sqd(z)
        x = dt / disc.t_eddy

        dvt = np.sqrt(- ve2 * np.expm1(-2*x))
        vt = np.exp(-x) * vt + dvt*self._rng.standard_normal(size=len(z))

        return z, vd, vt
        
    def _kick(self, dt, particles):
        z, vd, vt = particles
        disc = self.disc

        # Step 1: Update the dust velocity 
        ts = self.St * disc.rho_0 / (disc.Omega * disc.rho(z))

        x = dt / ts
        vd = vd*np.exp(-x) - np.expm1(-x)*(disc.v_z(z) + vt + disc.a_grav(z)*ts)

        return z, vd, vt
        
    def _take_step(self, particles, dt):
        return self._drift(dt/2, self._kick(dt, self._drift(dt/2, particles)))

        

class Thomson86Model(MonteCarloModel):
    """Update particles using the Thomson (1986) model.

    Either the normalized normalized or non-normalized (default) equations
    can be used.
    """
    def __init__(self, disc, St, normalized=True):
        MonteCarloModel.__init__(self, disc, St)
        self.normalized = normalized
        
    def _take_step(self, particles, dt):
        z, vd, vt = particles
        disc = self.disc
            
        # Step 0: Store the old normalization
        if self.normalized:
            ve2_0 = disc.v_turb_sqd(z)
            
        # Step 1: Update position
        z += vd * dt
                        
        # Step 2: Compute the deterministic part of the velocity update
        ve2 = disc.v_turb_sqd(z)
        Un  = disc.v_z(z)
        dve2 = disc.dv_turb_sqd_dz(z)
        dln_rho = disc.dlnrho_dz(z)
            
        t_e = disc.t_eddy
        if self.normalized:
            vt *= np.sqrt(ve2/ve2_0)
            dvt = t_e*(0.5*dve2 + ve2*dln_rho)
        else:
            dvt = t_e*(0.5*dve2*(1 + (vt*vt + vt*Un)/ve2) + ve2*dln_rho)
        vt = vt * np.exp(-dt/t_e) - dvt * np.expm1(-dt/t_e)
            
        # Step 3: Add the random part:
        vt += np.sqrt(-ve2*np.expm1(-2*dt/t_e)) * \
            self._rng.standard_normal(size=len(z))
                        
        # Step 4: Update the dust velocity 
        ts = self.St * disc.rho_0 / (disc.Omega * disc.rho(z))
            
        vd = ((vd + disc.a_grav(z)*dt)*ts + (vt + Un)*dt) / (ts + dt)

        # And we're done
        
        return z, vd, vt


def integrate_dust(mc_model,
                   Npart=100, dt=0.001,
                   t_init=10**3, t_sample=10**3, dt_sample=10.0):
    
    z, vd, vt = [np.zeros(Npart, dtype='f8') for i in range(3)]

    # Initial run to equilibrium
    Nstep = int(t_init/dt + 0.5)
    parts = mc_model([z, vd, vt], Nstep, dt)

    # Run to compute samples
    mc_model.setup_sampling(dt_sample)

    Nstep = int(t_sample/dt + 0.5)
    parts = mc_model(parts, Nstep, dt)

    return mc_model.get_samples()


def bin_samples(data, bins, nbootstrap=100):
    """Bin the samples using bootstrap to estimate the uncertainty"""
    
    draws = []
    rng = np.random.default_rng()
    for i in range(nbootstrap):
        idx = rng.integers(len(data), size=len(data))

        draws.append(np.histogram(data[idx], bins=bins, density=True)[0])

    draws = np.array(draws)
    return np.mean(draws, axis=0), np.std(draws, axis=0)
    
def run_and_plot(disc, St, Npart, ax, bins):
    
    models = [
        Thomson86Model(disc, St),
        Ormel18Model(disc, St),
        Laibe20Model(disc, St),
        Ciesla10Model(disc, St)
    ]     
    labels = ['This work', 'Ormel & Liu (2018)',
              'Laibe et al. (2020)', 'Ciesla (2010)']

    for model, label in zip(models, labels):
        samples = integrate_dust(model, dt=0.01, Npart=Npart,
                                 t_init=1e4, t_sample=1e5)
        z, vd = samples['z'], samples['vd']

        rho, std = bin_samples(z, bins)

        zc = (bins[1:] + bins[:-1])/2
        ax.errorbar(zc, rho, std, ls='none', marker='+', label=label)
        
    ax.set_xlabel(r'$z/H$')
    ax.set_yscale('log')    


def main(Npart):

    fig, axes = plt.subplots(1, 3, figsize=(10,4))

    bins_0  = np.linspace(-5,5,101)
    bins_St = np.linspace(-2,2,101)

    Sts = [0, 0, 5e-2, 1.0]
    Ws  = [0.3, 0.05, 0.05, 0.05]
    bins_ = [bins_0, bins_0, bins_St, bins_St]

    for St, W, bins, ax in zip(Sts, Ws, bins_, axes):
        disc = DiscModel(W=W, alpha_0=0.01)#, z_t=2**0.5)
        run_and_plot(disc, St, Npart, ax, bins)

        ax.text(bins[0]*0.95, 1.0, "$St={:.2g}$\n$W={:.2f}H$".format(St, W),
                verticalalignment='center')

        z = np.linspace(bins[0],bins[-1], 10**4)

        # Youdin & Lithwick (2007)
        Hp = disc.H / np.sqrt((1 + St/1e-2) * (1 + St/(1 + St)))
        rho_z = np.exp(-0.5*(z/Hp)**2) / (np.sqrt(2*np.pi)*Hp)
        ax.plot(z/disc.H, rho_z, c='k', zorder=10)

        ax.set_ylim(1e-4, 2)
            
    axes[1].legend(frameon=False)
    axes[0].set_ylabel('PDF')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_part", "-N", type=int, default=10**3)
    args = parser.parse_args()

    main(args.num_part)

    
# Plots:
# St = 0, W=0.3
# St = 0, W=0.1
# St = 0.01, W=0.1
# alpha(z)? Velocity distribution?
