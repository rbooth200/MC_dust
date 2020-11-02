#ifndef _HEADERS_DISC_H_
#define _HEADERS_DISC_H_

#include <boost/numeric/odeint.hpp>
#include <cmath>

#include "interpolate.h"

/* class SoundSpeedProfile
 *  Simple model for a sound speed profile that switches from one value to
 *  another over a narrow range. Uses Tanh interpolation between the two:
 *
 *     c_s(z) = 0.5*(cs0 + cs1) + 0.5*(cs1 - cs0) * np.tanh((z-z0)/W).
 *
 *  Parameters
 *  ----------
 *  cs0 : float
 *      Sound speed at z=0
 *  cs1 : float
 *      Sound speed at z=np.inf
 * z0 : float
 *      Location of the switch
 *  W : float
 *     Width of the transition region
 */
class SoundSpeedProfile {
   public:
    struct params {
        double cs0 = 1.0, cs1 = 10.0;
        double z_t = 3.0, W = 0.01;
    };

    SoundSpeedProfile() = default;
    SoundSpeedProfile(params p) : _p(p){};
    SoundSpeedProfile(double cs0, double cs1, double z_t, double W)
     : _p({cs0, cs1, z_t, W}){};

    /* Sound Speed, c_s at z */
    double operator()(double z) const {
        return 0.5 * (_p.cs0 + _p.cs1) +
               0.5 * (_p.cs1 - _p.cs0) *
                   std::tanh((std::abs(z) - _p.z_t) / _p.W);
    }
    /* First derivative of sound-speed, dc_s/dz at z */
    double deriv(double z) const {
        double cosh_z = std::cosh((std::abs(z) - _p.z_t) / _p.W);
        double _deriv = 0.5 * (_p.cs1 - _p.cs0) / (_p.W * cosh_z * cosh_z);

        return std::copysign(_deriv, z);
    }

    params get_params() const { return _p; }

    void set_params(params p) { _p = p; }

   private:
    params _p;
};

/* Model for the vertical profile of a disc.
 *
 *  Assumes z << R, hence the gravitational acceleration is - Omega^2*z.
 *
 *  Turbulence is parameterized using the same form as the sound-speed
 *  although the transition is specified sepearately:
 *       D(z) = 0.5*(D0 + D1) + 0.5*(D1 - D0) * np.tanh((z-z0)/W).
 *
 *  Parameters
 *  ----------
 *  c_s : Sound speed profile
 *      Provides sound speed and derivative at z
 *  rho_0 : float
 *      Gas density at z=0
 *  Omega : float
 *      Keplerian speed
 *  gravity : string, default = 'linear'
 *      How gravity is modelled:
 *           constant   : g = -Omega**2
 *           linear     : g = -Omega**2 * z
 *           no_gravity : g = 0
 *  v_0 : float
 *      Gas velocity at z=0
 *  D_0 : float,
 *      Mid-plane diffusion coefficient
 *  D_1 : float,
 *      Surface diffusion coefficient
 *  t_eddy : float
 *      Turbulence correlation time (eddy time)
 *  Zmax : float
 *      Maximum height to evaluate the properties up to.
 */
class DiscModel {
   public:
    enum class GRAVITY { constant, linear, full, no_gravity };
    struct params {
        double rho_0 = 1;
        double Omega = 1;
        double aspect = 0.05;
        GRAVITY gravity = GRAVITY::linear;
        double v_0 = 0;
        double D_0 = 0.01, D_1 = 0.01, t_eddy = 1.0;
        double z_t = 1.0, W = 1.0;
        double Zmax = 5;
    };

    DiscModel(SoundSpeedProfile c_s)
     : _cs(c_s), _diffusion(SoundSpeedProfile(_p.D_0, _p.D_1, _p.z_t, _p.W)) {
        _compute_structure();
    }

    DiscModel(SoundSpeedProfile c_s, params p)
     : _p(p),
       _cs(c_s),
       _diffusion(SoundSpeedProfile(p.D_0, p.D_1, p.z_t, p.W)) {
        _compute_structure();
    }

    double density(double z) const { return _density(std::abs(z)); }
    double Sigma(double z) const { return _density.integrate(0, std::abs(z)) ;}
    double dln_density_dz(double z) const {
        double rho = density(z);
        double v = _p.rho_0 * _p.v_0 / rho;

        /* Sound speed / pressure */
        double cs = _cs(z), dcs = _cs.deriv(z);

        /* Gravitational acceleration */
        double g = gravity(z);

        return (2 * cs * dcs - g) / (v * v - cs * cs);
    }
    double v_z(double z) const { return _p.v_0 * _p.rho_0 / density(z); }
    double sound_speed(double z) const { return _cs(z); }
    double gravity(double z) const {
        double R = _cs(0) / (_p.Omega * _p.aspect);
        double fac = 1 / std::sqrt(1 + z * z / (R * R));
        switch (_p.gravity) {
        case GRAVITY::constant:
            return -std::copysign(_p.Omega * _p.Omega, z);
        case GRAVITY::linear:
            return -_p.Omega * _p.Omega * z;
        case GRAVITY::full:
            return -_p.Omega * _p.Omega * z * fac * fac * fac;
        case GRAVITY::no_gravity:
            return 0;
        }
        return 0;
    }
    double v_turb_sqd(double z) const { return _diffusion(z) / _p.t_eddy; }
    double dv_turb_sqd_dz(double z) const {
        return _diffusion.deriv(z) / _p.t_eddy;
    }
    double t_eddy() const { return _p.t_eddy; }

    params disc_params() const { return _p; }
    SoundSpeedProfile::params sound_speed_params() const {
        return _cs.get_params();
    }
    SoundSpeedProfile::params turbulence_params() const {
        return _diffusion.get_params();
    }

   private:
    /* Evaluate the 1D-steady state hydrodynamics equations */
    class HydroEquations {
       public:
        typedef std::array<double, 1> state;

        HydroEquations(const SoundSpeedProfile& c_s, const params& p)
         : _p(p), _cs(c_s){};

        void operator()(const state& x, state& dxdz, double z) const {
            double rho = x[0];

            /* Continuity */
            double v = _p.rho_0 * _p.v_0 / rho;

            /* Sound speed / pressure */
            double cs = _cs(z), dcs = _cs.deriv(z);

            /* Gravitational acceleration */
            double g = 0;
            double R = _cs(0) / (_p.Omega * _p.aspect);
            double fac = 1 / std::sqrt(1 + z * z / (R * R));
            switch (_p.gravity) {
            case GRAVITY::constant:
                g = -std::copysign(_p.Omega * _p.Omega, z);
                break;
            case GRAVITY::linear:
                g = -_p.Omega * _p.Omega * z;
                break;
            case GRAVITY::full:
                g = -_p.Omega * _p.Omega * z * fac * fac * fac;
                break;
            case GRAVITY::no_gravity:
                g = 0;
                break;
            }
            dxdz[0] = rho * (2 * cs * dcs - g) / (v * v - cs * cs);
        }

       private:
        const params& _p;
        const SoundSpeedProfile& _cs;
    };

    void _compute_structure() {
        // Setup the ODE integrator
        using namespace boost::numeric::odeint;

        typedef runge_kutta_dopri5<typename HydroEquations::state>
            stepper_type;
        auto stepper = make_controlled(1e-10, 1e-10, stepper_type());

        // Initialize the integrator
        double z = 0;
        double dz = 1e-2 * _p.z_t;

        HydroEquations eqn(_cs, _p);
        HydroEquations::state state = {_p.rho_0};
        stepper.initialize(eqn, state, z);

        // Save the outputs for interpolation
        std::vector<double> z_i, rho_i;
        z_i.push_back(0), rho_i.push_back(_p.rho_0);

        int steps = 0;
        double z_t = _cs.get_params().z_t;
        do {
            if (steps >= MAX_STEPS)
                throw std::runtime_error("Failed to compute disc structure");
            steps++;

            // Ensure we don't cross the transition without at least one point
            // within it.
            if (z < z_t)
                dz = std::min(dz, z_t - z);
            else
                dz = std::min(dz, _p.Zmax - z);

            while (stepper.try_step(eqn, state, z, dz) == fail)
                ;

            z_i.push_back(z);
            rho_i.push_back(state[0]);
        } while (z < _p.Zmax);

        // Setup the interpolants:
        _density.set_data(z_i, rho_i);
    }
    params _p;
    SoundSpeedProfile _cs, _diffusion;

    PchipInterpolator<1> _density;
    static const int MAX_STEPS = 10000;
};
#endif  //_HEADERS_DISC_H_