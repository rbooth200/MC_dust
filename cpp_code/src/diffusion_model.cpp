
#include <cmath>

#include "block_tridiag_solve.h"
#include "diffusion_model.h" 


void DiffusionModel::solve_structure() {

    // Setup coefficient matrix:
    std::vector<double> l(_num_cells), d(_num_cells), u(_num_cells) ;

    // First cell has fixed density
    l[0] = 0 ;
    d[0] = 1 ;
    u[0] = 0 ;

    for (int i=1; i < _num_cells; i++) {
        double zl = ze(i), zr = ze(i+1) ;

        double rho_l = _disc.density(zl) ;
        double rho_r = _disc.density(zr) ;

        double vl = v_dust(zl) ;
        double vr = v_dust(zr) ;

        double Dl = _disc.v_turb_sqd(zl) * _disc.t_eddy() ;
        double Dr = _disc.v_turb_sqd(zr) * _disc.t_eddy() ;

        l[i] = 0.5*(vl   ) + Dl/(_dz*_disc.density(zc(i-1))) * (rho_l      ) ;
        d[i] = 0.5*(vl-vr) - Dl/(_dz*_disc.density(zc( i ))) * (rho_l+rho_r) ;
        u[i] = 0.5*(  -vr) + Dr/(_dz*_disc.density(zc(i+1))) * (     +rho_r) ;

        // Overwrite final cell, for which we have a special boundary condition
        if (i == _num_cells -1) {
            d[i] = 0.5*vl ;
            u[i] = 0 ;
        }
    }

    // For the outer boundary we use an advective flux with a correction for
    // the turbulent velocity.
    double v0 = v_dust(_Zmax) ; 
    double vt2 = _disc.v_turb_sqd(_Zmax) ;

    double v1 = std::sqrt(vt2/(2*M_PI)) * std::exp(-0.5*(v0*v0)/vt2) ;
    double v2 = 0.5*v0*std::erfc(-std::sqrt(2/vt2)*v0) ;
    std::cout << v0 << " " << std::sqrt(vt2) << "\n" ;
    std::cout << v1 << " " << v2 << "\n" ;

    d[_num_cells] -=  (v1 + v2) ;



    // Right hand side: set the density in the first cell to 1
    std::vector<double> rhs(_num_cells) ;
    rhs[0] = 1;

    // Solve the linear system
    BlockTriDiagSolver<1> solver(_num_cells) ;
    solver.factor_matrix(&(l[0]), &(d[0]), &(u[0])) ;

    solver.solve(&(rhs[0]), &(_rho[0])) ;
}