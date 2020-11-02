
#include <cmath>
#include <filesystem>
#include <fstream>
#include <tuple>

#include "disc.h"
#include "diffusion_model.h"
#include "logging.h"

static Logger _logger;



double find_input_velocity(double R, double z_IF) {

    // Compute the the ionization front height assuming:
    //   - The sound speed in the ionized gas is 12.85 km/s
    //   - The Mach number down-wind of the ionization front is 0.5
    //   - The disc structure is Gaussian

    double cs_ratio = 12.85 / (29.8 * 0.05 * std::pow(R, -0.25)) ;

    double M_disc = std::exp(-0.5 * z_IF * z_IF) /
        (cs_ratio*(1.25 + std::sqrt(25/16. - 1/(cs_ratio*cs_ratio)))) ;

    return M_disc ;
} ;

DiscModel build_disc(double alpha = 0.05,
                     double z_IF = 3.5) {
    double R = 10;  // Radius in AU

    double H_mid = 0.05 * std::pow(R, 0.25);
    // 12.85 (10^4 K in km/s, 29.8 is keplerian velocity at 1 au)
    double H_1 = (12.85 / 29.8) * std::pow(R, 0.5);

    SoundSpeedProfile::params pS;
    pS.W = 0.05 ;
    pS.z_t = 10 ;

    pS.cs0 = 1.0;
    pS.cs1 = 1.0;

    DiscModel::params p;
    p.D_0 = alpha;
    p.D_1 = alpha;
    p.z_t = 10 ;
    p.W = 0.05;

    p.aspect = H_mid;
    p.gravity = DiscModel::GRAVITY::full;

    p.Zmax = z_IF + 1;
    p.v_0 = find_input_velocity(R, z_IF) ;

    DiscModel disc(SoundSpeedProfile(pS), p);

    _logger << "Ionization front height = " << z_IF << " H\n"
            << "Aspect Ratio            = " << p.aspect << "\n"
            << "Sound-speed ratio       = " << H_1 / H_mid << "\n"
            << "Turbulent alpha param   = " << alpha << "\n"
            << "Mid plane velocity      = " << p.v_0 << "\n" ;

    double fac = std::pow(1 + std::pow(z_IF * p.aspect, 2), 1.5);
    double St_mid = fac * p.v_0 / (p.Omega * p.z_t);
    double St_crit = St_mid * p.rho_0 / disc.density(z_IF);
    double St_IF = St_crit * pS.cs1 / pS.cs0;
    _logger << "Critical Stokes number  = " << St_mid << " (z=0), " << St_crit
            << " (z=z_t), " << St_IF << " (wind)"
            << "\n";

    _logger.flush();

    return disc;
}



std::filesystem::path create_base_directory(double St, double alpha, 
                                            double z_IF) {
    std::stringstream ss;
    ss << "output_diffusion/z_IF" << z_IF << "/alpha" << alpha
       << "/St" << St;

    std::filesystem::path base = ss.str();
    return base;
}

std::tuple<double, double, double, int> parse_args(int argc, char* argv[]) {
    double St = 0;
    double alpha = 0.05;
    double z_IF = 3.5;
    int num_cells = 10000 ;

    bool keep_stdout = true;

    auto parse_double = [](std::string item) {
        std::stringstream ss(item);
        double val;
        ss >> val;
        return val;
    };
    auto parse_int = [](std::string item) {
        std::stringstream ss(item);
        int val;
        ss >> val;
        return val;
    };

    if (argc > 1) St = parse_double(argv[1]);
    if (argc > 2) alpha = parse_double(argv[2]);
    if (argc > 3) z_IF = parse_double(argv[3]);
    if (argc > 4) num_cells = parse_int(argv[4]);
    if (argc > 5) keep_stdout = parse_int(argv[5]);

    _logger = Logger(create_base_directory(St, alpha, z_IF) / "log.txt",
                     keep_stdout);

    return {St, alpha, z_IF, num_cells};
}

void save_results(std::filesystem::path DIR_NAME,
                  DiscModel& disc, DiffusionModel& dust) {

    std::filesystem::path filename = DIR_NAME / "results.dat" ;

    std::ofstream of(filename);

    int num_cells = dust.num_cells() ;

    of << "# z rho_d rho_g v_d v_g\n" ;
    for (int i=0; i < num_cells; i++) {
        double z = dust.zc(i) ;
        of << z << " " 
           << dust.density(i) << " "
           << disc.density(z) << " "
           << dust.v_dust(z) << " "
           << disc.v_z(z) << "\n" ;
    }
}

int main(int argc, char* argv[]) {

    int num_cells ;
    double St, alpha, z_IF;
    std::tie(St, alpha, z_IF, num_cells) = parse_args(argc, argv);

    std::filesystem::path BASE_DIR = create_base_directory(St, alpha, z_IF);
    std::filesystem::create_directories(BASE_DIR) ;

    DiscModel disc = build_disc(alpha, z_IF) ;
    DiffusionModel dust(disc, St, z_IF, num_cells) ;

    save_results(BASE_DIR, disc, dust) ;
} ;