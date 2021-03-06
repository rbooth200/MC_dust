

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>

#include "disc.h"
#include "histogram.h"
#include "logging.h"
#include "monte_carlo_model.h"

static Logger _logger;

double find_input_velocity(SoundSpeedProfile::params p_cs, DiscModel::params p,
                           double v_target) {
    double v0 = 0;
    double v1 = v_target;
    do {
        double vc = 0.5 * (v0 + v1);
        p.v_0 = vc;
        double vf;
        try {
            DiscModel disc(SoundSpeedProfile(p_cs), p);
            vf = disc.v_z(p_cs.z_t + 3 * p_cs.W);
        } catch (std::exception&) {
            // Velocity too high
            v1 = vc;
            continue;
        }
        if (vf >= v_target) v1 = vc;
        if (vf <= v_target) v0 = vc;

    } while ((v1 - v0) > 1e-12 * v1);

    return 0.5 * (v0 + v1);
}

DiscModel build_disc(double W = 0.001, double alpha = 0.05,
                     double z_IF = 3.5) {
    double R = 10;  // Radius in AU

    double H_mid = 0.05 * std::pow(R, 0.25);
    // 12.85 (10^4 K in km/s, 29.8 is keplerian velocity at 1 au)
    double H_1 = (12.85 / 29.8) * std::pow(R, 0.5);

    SoundSpeedProfile::params pS;
    pS.W = W / (3 * H_mid);  // Definition used in HC2020
    pS.z_t = z_IF;

    pS.cs0 = 1.0;
    pS.cs1 = H_1 / H_mid;

    DiscModel::params p;
    p.D_0 = alpha;
    p.D_1 = alpha;
    p.z_t = pS.z_t;
    p.W = 0.05;

    p.aspect = H_mid;
    p.gravity = DiscModel::GRAVITY::full;

    p.Zmax = 10;
    p.v_0 = find_input_velocity(pS, p, 0.5 * pS.cs1);

    DiscModel disc(SoundSpeedProfile(pS), p);

    double z_f = pS.z_t + 3 * p.W;
    _logger << "Ionization front height = " << p.z_t << " H\n"
            << "Ionization front width  = " << pS.W * 3 << " H\n"
            << "Aspect Ratio            = " << p.aspect << "\n"
            << "Sound-speed ratio       = " << pS.cs1 << "\n"
            << "Turbulent alpha param   = " << alpha << "\n"
            << "Mid plane velocity      = " << p.v_0 << "\n"
            << "Downstream Mach number  = " << disc.v_z(z_f) / pS.cs1 << "\n";

    double z_l = pS.z_t - 3 * p.W;
    double fac = std::pow(1 + std::pow(p.z_t * p.aspect, 2), 1.5);
    double St_mid = fac * p.v_0 / (p.Omega * p.z_t);
    double St_crit = St_mid * p.rho_0 / disc.density(z_l);
    double St_IF = St_crit * pS.cs1 / pS.cs0;
    _logger << "Critical Stokes number  = " << St_mid << " (z=0), " << St_crit
            << " (z=z_t), " << St_IF << " (wind)"
            << "\n";

    _logger.flush();

    return disc;
}

std::filesystem::path create_base_directory(double St, double W, double alpha,
                                            double z_IF) {
    std::stringstream ss;
    ss << "output/z_IF" << z_IF << "/alpha" << alpha << "/W" << W << "/St"
       << St;

    std::filesystem::path base = ss.str();
    return base;
}

int get_num_snaps(std::filesystem::path SNAP_DIR) {
    int num_snaps = 0;

    for (auto& f_iter : std::filesystem::directory_iterator(SNAP_DIR))
        if (f_iter.is_regular_file()) {
            std::string f = f_iter.path();
            if (f.find(".dat") + 4 == f.length()) num_snaps++;
        }
    return num_snaps;
}

std::tuple<double, double, double, double> parse_args(int argc, char* argv[]) {
    double St = 0;
    double W = 1e-3;
    double alpha = 0.05;
    double z_IF = 3.5;

    bool keep_stdout = true;

    auto parse_double = [](std::string item) {
        std::stringstream ss(item);
        double val;
        ss >> val;
        return val;
    };
    auto parse_bool = [](std::string item) {
        std::stringstream ss(item);
        int val;
        ss >> val;
        return val;
    };

    if (argc > 1) St = parse_double(argv[1]);
    if (argc > 2) W = parse_double(argv[2]);
    if (argc > 3) alpha = parse_double(argv[3]);
    if (argc > 4) z_IF = parse_double(argv[4]);
    if (argc > 5) keep_stdout = parse_bool(argv[5]);

    _logger = Logger(create_base_directory(St, W, alpha, z_IF) / "log.txt",
                     keep_stdout);

    return {St, W, alpha, z_IF};
}

int main(int argc, char* argv[]) {
    // Load the parameters
    double St, W, alpha, z_IF;
    std::tie(St, W, alpha, z_IF) = parse_args(argc, argv);

    std::filesystem::path BASE_DIR = create_base_directory(St, W, alpha, z_IF);
    std::filesystem::path HIST_DIR = (BASE_DIR / "binned");
    std::filesystem::path SNAP_DIR = (BASE_DIR / "snap");

    std::filesystem::create_directories(HIST_DIR);
    std::filesystem::create_directories(SNAP_DIR);

    // For storing the results
    HistogramCounts hist({0, 5}, 5000);

    auto disc = build_disc(W, alpha, z_IF);
    _logger << "Particle Stokes number  = " << St << "\n\n";

    // Monte Carlo model
    ConstantInjectionRate injection(4., 0., disc.v_z(0),
                                    std::sqrt(disc.v_turb_sqd(0)));
    Boundary boundary({0, 5}, Boundary::TYPE::reflecting,
                      Boundary::TYPE::outflow);

    Thomson1986Model<> thomson_method(St, disc);
    MonteCarloModel<Thomson1986Model<>, ConstantInjectionRate> mc_model(
        thomson_method, boundary, injection);

    // Particles
    Particles p;
    for (int i = 0; i < 16; i++) p.add_particle(0, 0, 0);

    // Time integration
    double dt_sample = 1;

    double t_sample = 1e5;
    _logger << "Computing Samples"
            << "\n";
    _logger.flush();

    // Restart from last snapshot:
    int count = 1000 * get_num_snaps(SNAP_DIR);
    double t = dt_sample * count;

    if (count > 0) {
        std::stringstream fname;
        fname << "snap_" << count / 1000 << ".dat";
        std::filesystem::path in = (SNAP_DIR / fname.str());
        p.read_binary(in);
    }

    // Samples
    _logger << "\rProgress=" << 100 * t / t_sample << "%, "
            << "num_part=" << p.size;
    _logger.flush();

    do {
        mc_model(p, dt_sample);
        hist.add_sample(p);
        t += dt_sample;

        count += 1;
        if ((count % 100) == 0) {
            /* Report progress */
            _logger << "\rProgress=" << 100 * t / t_sample << "%, "
                    << "num_part=" << p.size << ", "
                    << "dt=" << p.get_current_timestep(dt_sample)
                    << ", num_active=" << p.Nactive << ", max(z)="
                    << *std::max_element(p.z.begin(), p.z.end());
            _logger.flush();
        }
        if ((count % 1000) == 0) {
            /* Store the statistics */
            auto stats = hist.compute_stats();
            std::stringstream stat_fname;
            stat_fname << "hist_" << count / 1000 << ".dat";
            std::filesystem::path stat_out = (HIST_DIR / stat_fname.str());

            std::ofstream of(stat_out);

            of << "# z count mean(v) std(v) rho_gas v_gas v_dust\n";
            for (int i = 0; i < stats.nbins; i++) {
                double t_s = St / (disc.density(stats.centres[i]) *
                                   disc.sound_speed(stats.centres[i]));
                of << stats.centres[i] << " " << stats.counts[i] << " "
                   << stats.mean_v[i] << " " << stats.std_v[i] << " "
                   << disc.density(stats.centres[i]) << " "
                   << disc.v_z(stats.centres[i]) << " "
                   << disc.v_z(stats.centres[i]) +
                          disc.gravity(stats.centres[i]) * t_s
                   << "\n";
            }
            hist.reset();

            /* Dump the particle data */
            std::stringstream part_fname;
            part_fname << "snap_" << count / 1000 << ".dat";
            std::filesystem::path part_out = (SNAP_DIR / part_fname.str());

            p.write_binary(t, part_out);
        }

    } while (t < t_sample);
    _logger << "\rProgress=" << 100 << "%,"
            << "num_part=" << p.size << "\n";
    _logger.flush();

    return 0;
}
