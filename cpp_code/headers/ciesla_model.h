#ifndef _HEADERS_CIESLA_MODEL_H_
#define _HEADERS_CIESLA_MODEL_H_

#include "disc.h"
#include "particles.h"

class Ciesla2010Model {
   public:
    struct timestep_params {
        double C_Omega = 0.01, C_grad = 0.05;
    };

    Ciesla2010Model(double ts0, DiscModel disc)
     : _disc(disc), _drag_coeff(ts0 * disc.density(0) * disc.sound_speed(0)){};

    void set_timestep_params(timestep_params p) { _p = p; }
    timestep_params get_timeste_params() const { return _p; }

    template <class RNG, class Particles>
    void take_step(Particles& p, double dt_max, RNG& rng) const {
        double t_e = _disc.t_eddy();

        std::normal_distribution<double> normal;

        for (int i = 0; i < p.Nactive; i++) {
            double z = p.z[i];

            double dt = p.get_level_timestep(dt_max, p.level[i]);

            double ts =
                _drag_coeff / (_disc.density(z) * _disc.sound_speed(z));

            double D = _disc.v_turb_sqd(z) * t_e;

            double Un = _disc.v_z(z);
            double Ufp =
                D * _disc.dln_density_dz(z) + _disc.dv_turb_sqd_dz(z) * t_e;
            double Ug = _disc.gravity(z) * ts;

            double zp = z + 0.5 * dt * Ufp;
            D = _disc.v_turb_sqd(zp) * t_e;

            double vt = std::sqrt(2 * D / dt) * normal(rng);
            double v = Un + Ufp + Ug + vt;
            z += v * dt;

            // Store the results
            p.z[i] = z;
            p.v[i] = v;
            p.v_turb[i] = vt;
        }
    }

    template<class Particles>
    void set_timestep_levels(Particles& p, double time_step) const {
        double dt_max = _p.C_Omega / _disc.disc_params().Omega;

        for (int i = 0; i < p.Nactive; i++) {
            double z = p.z[i];

            /* Time to diffuse across a gradient */
            double D = _disc.v_turb_sqd(z);
            double grad =
                D * _disc.dln_density_dz(z) + _disc.dv_turb_sqd_dz(z);
            double l = D / grad;

            double dt_1 = _p.C_grad * (l * l / D);

            double dt = 1 / (1 / dt_max + 1 / dt_1);

            p.set_timestep_level(i, dt, time_step);
        }
    };

   private:
    DiscModel _disc;
    timestep_params _p;
    double _drag_coeff;
};

#endif//_HEADERS_CIESLA_MODEL_H_
