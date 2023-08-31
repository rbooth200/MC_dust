#ifndef _HEADERS_MONTE_CARLO_MODEL_H_
#define _HEADERS_MONTE_CARLO_MODEL_H_

#include <random>
#include <vector>

#include "disc.h"
#include "domain.h"
#include "particles.h"

#include "ciesla_model.h"
#include "thomson_model.h"

class NoParticleInjection {
   public:
    template <class RNG, class Particles>
    int operator()(Particles& /*p*/, double /*dt*/, RNG&) {
        return 0;
    }
};

class ConstantInjectionRate {
   public:
    ConstantInjectionRate(double Ndot, double z0, double v0, double v_turb)
     :  _Ndot(Ndot), _z0(z0), _v0(v0), _v_turb(v_turb), _N(0) {}

    template <class RNG, class Particles>
    int operator()(Particles& p, double dt, RNG& rng) {
        _N += _Ndot * dt;

        int n_inj = _N + 0.5;
        if (n_inj > 0) {
            std::normal_distribution<double> v_dist(0, _v_turb) ;
            for (int i = 0; i < n_inj; i++) {
                p.add_particle(_z0, _v0, v_dist(rng));
            }
        }
        _N -= n_inj;

        return n_inj;
    }

   private:
    double _Ndot, _z0, _v0, _v_turb, _N;
};

/* class Boundary
 *
 * Applies the boundary conditions for each particle.
 *
 * Boundary types supported are:
 *   - open :
 *        No boundary, domain is infinite in that direction
 *   - reflecting :
 *        Particles crossing the boundary will have the sign of their velocity
 *        flipped and their position reflected.
 *   - ouflow :
 *        Particles crossing the boundary will be flagged as dead.
 */
class Boundary {
   public:
    enum class TYPE { open, reflecting, outflow };

    Boundary(){};

    Boundary(Domain domain, TYPE left_boundary, TYPE right_boundary)
     : _domain(domain), _left(left_boundary), _right(right_boundary) {}

    template<class Particles>
    bool operator()(Particles& p) {
        int size = p.size;
        bool dead_particles = false;
        for (int i = 0; i < size; i++) {
            switch (_left) {
            case TYPE::reflecting:
                if (p.z[i] < _domain.left_edge)
                    _reflect(p, i, _domain.left_edge);
                break;

            case TYPE::outflow:
                if (p.z[i] < _domain.left_edge) {
                    _outflow(p, i);
                    dead_particles = true;
                }
                break;
            case TYPE::open:
                break;
            }

            switch (_right) {
            case TYPE::reflecting:
                if (p.z[i] > _domain.right_edge)
                    _reflect(p, i, _domain.right_edge);
                break;
            case TYPE::outflow:
                if (p.z[i] > _domain.right_edge) {
                    _outflow(p, i);
                    dead_particles = true;
                }
                break;
            case TYPE::open:
                break;
            }
        }
        return dead_particles;
    }

   private:
    template<class Particles>
    void _reflect(Particles& p, int i, double z0) const {
        p.z[i] = 2 * z0 - p.z[i];
        p.v[i] *= -1;
        p.v_turb[i] *= -1;
    }
    template<class Particles>
    void _outflow(Particles& p, int i) const {
        p.level[i] = Particles::flags::dead;
    }

    Domain _domain = {0, 0};
    TYPE _left = TYPE::open;
    TYPE _right = TYPE::open;
};

class UserLoopMethod {
   public:

    template <class RNG, class Particles>
    void operator()(Particles& /*p*/, double /*dt*/, RNG&) {
    };
} ;

template <class Method, 
          class InjectionMethod = NoParticleInjection, 
          class UserLoopMethod = UserLoopMethod>
class MonteCarloModel {
   public:
    MonteCarloModel(Method MCmethod, Boundary boundary,
                    InjectionMethod injection = InjectionMethod(),
                    UserLoopMethod user_loop_method = UserLoopMethod(),
                    std::mt19937 rng = std::mt19937())
     : _rng(rng),
       _MCmethod(MCmethod),
       _boundary(boundary),
       _particle_injection(injection),
       _user_loop_method(user_loop_method){};

    MonteCarloModel(Method method, InjectionMethod injection,
                    std::mt19937 rng = std::mt19937())
     : _rng(rng), _MCmethod(method), _particle_injection(injection){};

    MonteCarloModel(Method method, std::mt19937 rng = std::mt19937())
     : _rng(rng), _MCmethod(method){};

    template<class Particles>
    void operator()(Particles& p, double tmax) {
        p.reset_time();
        _MCmethod.set_timestep_levels(p, tmax);  // Initialize the timesteps
        do {
            p.set_active_particles();

            _MCmethod.set_timestep_levels(p, tmax);

            p.update_timestep_bins();

            // Inject particles
            double dt = p.get_current_timestep(tmax);
            _particle_injection(p, dt, _rng);

            _MCmethod.take_step(p, tmax, _rng);

            // Apply user-supplied methods:
            _user_loop_method(p, dt, _rng);

            // Apply boundary conditions and remove particles that
            // have left the domain.
            bool dead_particles = _boundary(p);
            if (dead_particles) p.delete_dead_particles();

            p.increment_time();

        } while (p.get_current_time(tmax) < tmax);
    }

   private:
    std::mt19937 _rng;
    Method _MCmethod;
    Boundary _boundary;
    InjectionMethod _particle_injection;
    UserLoopMethod _user_loop_method;
};


#endif  //_HEADERS_MONTE_CARLO_MODEL_H_
