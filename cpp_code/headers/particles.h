#ifndef _HEADERS_PARICLES_H_
#define _HEADERS_PARICLES_H_

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "utils.h"

class Particles {
   public:
    enum flags { dead = -1 };

    void set_active_particles() {
        int current_level = min_active_level();

        if (current_level > 0)
            Nactive = level_start[current_level - 1];
        else
            Nactive = size;
    };

    void set_timestep_level(int part_id, double dt, double dt_max) {
        int level_i = 0;
        while (dt < dt_max) {
            dt *= 2;
            level_i++;
        }

        // Make sure the step is syncrhonised
        level_i = std::max(level_i, min_active_level());
        level[part_id] = level_i;
    }

    void update_timestep_bins() { sort_particles_by_level(); }

    double get_current_timestep(double tmax) const {
        return get_level_timestep(tmax, num_levels - 1);
    }

    double get_level_timestep(double tmax, int level) const {
        return tmax / (uint64_t(1) << level);
    }

    double get_current_time(double tmax) const {
        return current_integer_time * tmax / (uint64_t(1) << max_level);
    }

    void increment_time() {
        for (int i = 0; i < Nactive; i++) {
            if (time[i] != current_integer_time) {
                std::cout << "Error, particle not synchonized!\n";
                std::cout << "i=" << i << " Nactive=" << Nactive << "\n"
                          << "time[i]=" << time[i]
                          << " current time=" << current_integer_time << ","
                          << " difference="
                          << int64_t(current_integer_time - time[i]) << ". "
                          << " current step="
                          << (uint64_t(1) << (max_level + 1 - num_levels))
                          << "\n";
                std::cout << "level[i]=" << level[i]
                          << " num_levels=" << num_levels
                          << " min active level=" << min_active_level()
                          << "\n";
                assert(time[i] == current_integer_time);
            }
            time[i] += (uint64_t(1) << (max_level - level[i]));
        }
        current_integer_time += (uint64_t(1) << (max_level + 1 - num_levels));
    }

    void reset_time() {
        current_integer_time = 0;
        for (int i = 0; i < size; i++) time[i] = 0;

        Nactive = size;
    }

    bool is_active(int level) const {
        uint64_t step_length = (uint64_t(1) << (max_level - level));
        return (current_integer_time % step_length) == 0;
    }

    int min_active_level() {
        int level = 0;
        while (not is_active(level)) level++;
        return level;
    }

    void add_particle(double zi, double vi, double vt) {
        z.push_back(zi);
        v.push_back(vi);
        v_turb.push_back(vt);
        time.push_back(current_integer_time);

        level.push_back(num_levels - 1);
        ++size;

        // Move the particle into right level.
        int p = size - 1;
        for (int l = 0; l < num_levels - 1; l++) {
            int q = level_start[l];
            level_start[l]++;
            swap(p, q);
            p = q;
        }
        swap(p, 0);

        for (int i = 1; i < size; i++) {
            assert(level[i - 1] >= level[i]);
        }
        set_active_particles();
    }

    void delete_dead_particles() {
        int i = 0;
        while (i < Nactive) {
            if (level[i] != flags::dead) {
                i++;
                continue;
            } else {
                // Move the dead particle to the end, keeping the particles
                // ordered.
                int l = 0;
                while (l < num_levels) {
                    int q = std::max(level_start[l] - 1, i);

                    swap(q, size - 1);
                    while (level_start[l] ==
                           q + 1)  // Account for empty levels
                        level_start[l++]--;

                    if (q == i) {  // We've found the dead particle
                        --size;
                        --Nactive;

                        break;
                    }
                }
            }
        }
        if (size < static_cast<int>(z.size())) {
            z.resize(size);
            v.resize(size);
            v_turb.resize(size);
            level.resize(size);
            time.resize(size);

            for (int i = 1; i < size; i++) {
                assert(level[i - 1] >= level[i]);
            }
        }
        set_active_particles();
    }

    void write_ASCII(double time, std::ostream& f) const {
        f.precision(16);
        f << "# Monte-Carlo Particle Snapshot, ASCII v1\n";
        f << "# Time=" << time << "\n";

        f << "# z v v_turb level\n";
        for (int i = 0; i < size; i++)
            f << z[i] << " " << v[i] << " " << v_turb[i] << " " << level[i]
              << "\n";
    }

    void write_binary(double time, std::ostream& f) const {
        auto write_double = [&f](double x) {
            f.write(reinterpret_cast<char*>(&x), sizeof(double));
        };
        auto write_int = [&f](int x) {
            f.write(reinterpret_cast<char*>(&x), sizeof(int));
        };

        f << "# Monte-Carlo Particle Snapshot, binary v1\n";
        write_double(time);
        write_int(size);
        for (int i = 0; i < size; i++) {
            write_double(z[i]);
            write_double(v[i]);
            write_double(v_turb[i]);
            write_int(level[i]);
        }
    }

    double read_ASCII(std::istream& f) {
        std::string line;
        getline(f, line);

        if (line != "# Monte-Carlo Particle Snapshot, ASCII v1")
            throw std::invalid_argument(
                "file is not an ASCII particle snapshot");

        auto parse_item = [](std::string item, std::string type) {
            auto split_line = split(item, "=");
            if (split_line[0] != type)
                throw std::invalid_argument("Item type does not match");

            std::stringstream ss(split_line[1]);
            double val;
            ss >> val;
            return val;
        };

        getline(f, line);
        double sim_time = parse_item(line, "# Time");

        getline(f, line);
        if (line != "# z v v_turb level")
            throw std::invalid_argument("data types do not mathc");

        *this = Particles();
        do {
            getline(f, line);
            if (line.empty()) break;

            double zi, vi, vti;
            int li;

            std::stringstream ss(line);
            ss >> zi >> vi >> vti >> li;

            z.push_back(zi);
            v.push_back(vi);
            v_turb.push_back(vti);
            level.push_back(li);
            time.push_back(current_integer_time);
            ++size;
        } while (true);

        return sim_time;
    }

    double read_binary(std::istream& f) {
        auto read_double = [&f](double& x) {
            f.read(reinterpret_cast<char*>(&x), sizeof(double));
        };
        auto read_int = [&f](int& x) {
            f.read(reinterpret_cast<char*>(&x), sizeof(int));
        };

        std::string line;
        getline(f, line);

        if (line != "# Monte-Carlo Particle Snapshot, binary v1")
            throw std::invalid_argument(
                "file is not a binary particle snapshot");

        *this = Particles();

        double sim_time;
        read_double(sim_time);
        read_int(size);

        z.resize(size);
        v.resize(size);
        v_turb.resize(size);
        level.resize(size);
        time.resize(size);
        for (int i = 0; i < size; i++) {
            read_double(z[i]);
            read_double(v[i]);
            read_double(v_turb[i]);
            read_int(level[i]);
            time[i] = current_integer_time;
        }

        return sim_time;
    }

    void write_ASCII(double time, std::string filename) const {
        std::ofstream f(filename);
        write_ASCII(time, f);
    }
    void write_binary(double time, std::string filename) const {
        std::ofstream f(filename);
        write_binary(time, f);
    }
    double read_ASCII(std::string filename) {
        std::ifstream f(filename);
        return read_ASCII(f);
    }
    double read_binary(std::string filename) {
        std::ifstream f(filename);
        return read_binary(f);
    }

    std::vector<double> z, v, v_turb;
    std::vector<int> level;
    std::vector<uint64_t> time;
    int size = 0;
    int Nactive = 0;

   private:
    void swap(int i, int j) {
        std::swap(z[i], z[j]);
        std::swap(v[i], v[j]);
        std::swap(v_turb[i], v_turb[j]);
        std::swap(level[i], level[j]);
        std::swap(time[i], time[j]);
    }

    void sort_particles_by_level() {
        // Minimum level that we need to update
        int lmin = std::max(0, min_active_level() - 1);

        if (Nactive == 0) return;

        // First work out the number of particles in each level
        num_levels = 0;
        std::vector<int> num_in_level(num_levels, 0);
        for (int i = 0; i < Nactive; i++) {
            int l = level[i];

            // Inactive particles are sorted to the start of the inactive
            // particle list
            l = std::max(l, lmin);

            // Make sure we have enough storage
            if (l >= num_levels) {
                num_in_level.resize(l + 1);
                while (num_levels <= l) num_in_level[num_levels++] = 0;
            }
            num_in_level[l]++;
        }

        if (num_levels > max_level + 1)
            throw std::runtime_error("Maximum number of levels exceeded");

        // Find the starting point for each level
        level_start.resize(num_levels);
        level_start.back() = 0;
        for (int l = num_levels - 1; l > lmin; l--)
            level_start[l - 1] = level_start[l] + num_in_level[l];

        // Now sort the particles by level
        int p = 0, l = num_levels - 1;
        std::vector<int> start = level_start;
        while (p < Nactive) {
            // We've got all particles on this level, so move to the next.
            while (num_in_level[l] == 0 and l > lmin) {
                l--;
                p = start[l];
            }

            int lp = std::max(level[p], lmin);

            // If the particle is in the right level, move on
            if (lp == l) {
                p++;
                start[l]++;
                num_in_level[l]--;
            } else {
                // Find the next free space for this particle
                int q = start[lp];

                assert(num_in_level[lp] > 0);
                while (level[q] == lp) {
                    q++;
                    start[lp]++;
                    num_in_level[lp]--;
                }
                // Swap it into place
                swap(p, q);
            }
        }

        for (int l = lmin; l < num_levels; l++) assert(num_in_level[l] == 0);
    }
    static constexpr int max_level = 63;
    std::vector<int> level_start;
    uint64_t current_integer_time = 0;
    int num_levels = 0;
};

#endif  //_HEADERS_PARICLES_H_
