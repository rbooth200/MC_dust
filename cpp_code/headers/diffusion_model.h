#ifndef _DIFFUSION_MODEL_H_
#define _DIFFUSION_MODEL_H_

#include <vector>

#include "disc.h"

class DiffusionModel {

  public:
    DiffusionModel(DiscModel disc, double ts0, double Zmax, int num_cells)
      : _disc(disc), _ts0(ts0), 
        _drag_coeff(ts0 * disc.density(0) * disc.sound_speed(0)),
        _rho(num_cells), _Zmax(Zmax), _dz(Zmax/num_cells), 
        _num_cells(num_cells)
    {
        solve_structure() ;
    } 

    void solve_structure(DiscModel disc, double ts0) {
        _disc = disc ;
        _ts0 = ts0 ;
        solve_structure() ;
    }

    double v_dust(double z) const {
        double ts = _drag_coeff / (_disc.density(z) * _disc.sound_speed(z));
        return _disc.v_z(z) - _disc.gravity(z) * ts ;
    }

    double density(double i) const {
        return _rho[i] ;
    }
    double ze(double i) const {
        return _dz*i ;
    }
    double zc(double i) const {
        return _dz*(i+0.5) ;
    }
    int num_cells() const {
        return _num_cells ;
    }

  private:
    void solve_structure() ;

    DiscModel _disc ;
    double _ts0, _drag_coeff ;

    std::vector<double> _z, _rho ;
    double _Zmax, _dz ;
    int _num_cells ;
} ;

#endif// _DIFFUSION_MODEL_H_