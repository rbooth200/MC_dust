#ifndef _HEADERS_THOMSON_MODEL_H_
#define _HEADERS_THOMSON_MODEL_H_

#include "disc.h"
#include "particles.h"

class BaseModel {
  public:
    struct timestep_params {
        double C_eddy = 0.01, C_grad = 0.01, C_courant = 0.05;
    };

    BaseModel(double ts0, DiscModel disc)
     : _disc(disc), _drag_coeff(ts0 * disc.density(0) * disc.sound_speed(0)){};

    void set_timestep_params(timestep_params p) { _p = p; }
    timestep_params get_timestep_params() const { return _p; }

    template<class Particles>
    void set_timestep_levels(Particles& p, double time_step) const {
        double dt_max = _p.C_eddy * _disc.t_eddy();

        auto cs_par = _disc.sound_speed_params();
        auto D_par = _disc.turbulence_params();
        for (int i = 0; i < p.Nactive; i++) {
            double z = p.z[i];

            double v_e = std::sqrt(_disc.v_turb_sqd(z));
            double v = std::max(v_e, std::abs(p.v[i]));

            double dt_1 =
                _p.C_grad * 2 * v_e / std::abs(_disc.dv_turb_sqd_dz(z));
            double dt_2 =
                _p.C_grad / (v_e * std::abs(_disc.dln_density_dz(z)));

            double dz_cs = std::max(std::abs(z - cs_par.z_t), cs_par.W);
            double dz_D = std::max(std::abs(z - D_par.z_t), D_par.W);

            double dt_3 = _p.C_courant * std::min(dz_cs, dz_D) / v;

            double dt = 1 / (1 / dt_max + 1 / dt_1 + 1 / dt_2 + 1 / dt_3);

            p.set_timestep_level(i, dt, time_step);
        }
    };

  protected:
    DiscModel _disc;
    timestep_params _p;
    double _drag_coeff;
} ;


template <bool normalised = true>
class Thomson1986Model : public BaseModel { 
   public:
    using BaseModel::timestep_params ;
    using BaseModel::get_timestep_params ;
    using BaseModel::set_timestep_params ;
    using BaseModel::set_timestep_levels ;

    Thomson1986Model(double ts0, DiscModel disc)
     : BaseModel(ts0, disc){};

    template <class RNG, class Particles>
    void take_step(Particles& p, double dt_max, RNG& rng) const {
        double t_e = _disc.t_eddy();
        std::normal_distribution<double> normal;

        for (int i = 0; i < p.Nactive; i++) {
            double z = p.z[i], v = p.v[i], vt = p.v_turb[i];

            double dt = p.get_level_timestep(dt_max, p.level[i]);
            double exp_dt_te = std::exp(-dt / t_e);
            double expm1_dt_te = std::expm1(-dt / t_e);
            double expm1_2dt_te = std::expm1(-2 * dt / t_e);

            double ve2_0 = _disc.v_turb_sqd(z);

            // Step 1: Update the position
            z += v * dt;

            // Step 2: Compute the deterministic part of the turbulent
            //         velocity update
            double ve2 = _disc.v_turb_sqd(z);
            double Un = _disc.v_z(z);
            double dve2 = _disc.dv_turb_sqd_dz(z);
            double dln_rho = _disc.dln_density_dz(z);

            double dvt;
            if (normalised) {
                vt *= std::sqrt(ve2 / ve2_0);
                dvt = t_e * (0.5 * dve2 + ve2 * dln_rho);
            } else {
                dvt = t_e * (0.5 * dve2 * (1 + (vt * vt + vt * Un) / ve2) +
                             ve2 * dln_rho);
            }
            vt = vt * exp_dt_te - dvt * expm1_dt_te;

            // Step 3: Add the random part
            vt += std::sqrt(-ve2 * expm1_2dt_te) * normal(rng);

            // Step 4: Update the dust velocity
            double ts =
                _drag_coeff / (_disc.density(z) * _disc.sound_speed(z));
            v = ((v + _disc.gravity(z) * dt) * ts + (vt + Un) * dt) /
                (ts + dt);

            // Store the results
            p.z[i] = z;
            p.v[i] = v;
            p.v_turb[i] = vt;
        }
    }
};

template<bool SCA=false>
class Ormel2018Model : public BaseModel {
  public:
    using BaseModel::timestep_params ;
    using BaseModel::get_timestep_params ;
    using BaseModel::set_timestep_params ;
    using BaseModel::set_timestep_levels ;

    Ormel2018Model(double ts0, DiscModel disc)
     : BaseModel(ts0, disc){};

    template <class RNG, class Particles>
    void take_step(Particles& p, double dt_max, RNG& rng) const {
        double t_e = _disc.t_eddy();
        std::normal_distribution<double> normal;

        for (int i = 0; i < p.Nactive; i++) {
            double z = p.z[i], v = p.v[i], vt = p.v_turb[i];
            double dt = p.get_level_timestep(dt_max, p.level[i]);

            double vt2  = _disc.v_turb_sqd(z) ;
            double Un = _disc.v_z(z) ;
            double dvt2 = _disc.dv_turb_sqd_dz(z) ;
            
            double v_hs = (0.5*dvt2 + vt2*_disc.dln_density_dz(z))*t_e ;

            double gravity = _disc.gravity(z) ;
            double ts =
                _drag_coeff / (_disc.density(z) * _disc.sound_speed(z));

            z += v*dt ;
            if (SCA) {
                vt = std::sqrt(2*vt2*t_e/dt) * normal(rng) ;
                v  = gravity * ts + Un + v_hs + 0.5*dvt2*t_e/(1+ts/t_e) + vt ;
            } else { 
                double vg = Un + v_hs + vt ;
                v = ((v + gravity*dt)*ts + vg*dt) / (ts + dt) ;

                // Using integrated update from Laibe+ (2020)
                double x = dt / t_e ;
                double dvt = std::sqrt(-std::expm1(-2*x)*vt2) ;

                vt = std::exp(-x)*vt + dvt*normal(rng) ;
            }

            // Store the results
            p.z[i] = z;
            p.v[i] = v;
            p.v_turb[i] = vt;
        }
    }  
} ;

class Laibe2020Model : public BaseModel { 
   public:
    using BaseModel::timestep_params ;
    using BaseModel::get_timestep_params ;
    using BaseModel::set_timestep_params ;
    using BaseModel::set_timestep_levels ;

    Laibe2020Model(double ts0, DiscModel disc)
     : BaseModel(ts0, disc){};

    template <class RNG, class Particles>
    void take_step(Particles& p, double dt_max, RNG& rng) const {
        _drift(p, dt_max/2, rng) ;
        _kick(p, dt_max) ;
        _drift(p, dt_max/2, rng) ;
    }

   private:
    template <class RNG, class Particles>
    void _drift(Particles& p, double dt_max, RNG& rng) const {
        double t_e = _disc.t_eddy();
        std::normal_distribution<double> normal;

        for (int i = 0; i < p.Nactive; i++) {
            double z = p.z[i], v = p.v[i], vt = p.v_turb[i];
            double dt = p.get_level_timestep(dt_max, p.level[i]);

            // Step 0: Save current velocity
            double ve2_0 = _disc.v_turb_sqd(z) ;

            // Step 1: Move the particles         
            z += v * dt ;

            // Step 2: Update turbulent velocity
            double ve2_1 = _disc.v_turb_sqd(z) ;
            vt *= std::sqrt(ve2_1/ve2_0) ;

            vt = vt * std::exp(- dt / t_e) + 
                std::sqrt(-ve2_1 * std::expm1(-2 * dt / t_e)) * normal(rng) ;

            // Store the results
            p.z[i] = z;
            p.v[i] = v;
            p.v_turb[i] = vt;
        }       
    }
    
    template<class Particles>
    void _kick(Particles& p, double dt_max) const {
        for (int i = 0; i < p.Nactive; i++) {
            double z = p.z[i], v = p.v[i], vt = p.v_turb[i];
            double dt = p.get_level_timestep(dt_max, p.level[i]);

            double ts =
                _drag_coeff / (_disc.density(z) * _disc.sound_speed(z));

            double x = dt / ts ;
            v = v*std::exp(-x) 
                - std::expm1(-x)*(_disc.v_z(z) + vt + _disc.gravity(z)*ts) ;
            
            // Store the results
            p.z[i] = z;
            p.v[i] = v;
            p.v_turb[i] = vt;
        }      
    }
};


#endif//_HEADERS_THOMSON_MODEL_H_