#ifndef _HEADERS_HISTOGRAM_H_
#define _HEADERS_HISTOGRAM_H_

#include <cmath>
#include <vector>

#include "domain.h"
#include "particles.h"

class HistogramCounts {
   public:
    struct HistData {
        std::vector<double> edges, centres;
        std::vector<int> counts;
        std::vector<double> mean_v, std_v;
        int nbins;
    };

    HistogramCounts(Domain domain, int nbins)
     : counts(nbins), v(nbins), v_sqd(nbins), _domain(domain), _nbins(nbins){};

    template<class Particles>
    void add_sample(Particles& p) {
        double z0 = _domain.left_edge,
               dz = (_domain.right_edge - _domain.left_edge) / _nbins;

        for (int i = 0; i < p.size; i++) {
            int bin = (p.z[i] - z0) / dz;

            if (bin >= 0 and bin < _nbins) {
                counts[bin]++;
                v[bin] += p.v[i];
                v_sqd[bin] += p.v[i] * p.v[i];
            }
        }
    }

    void reset() {
        counts = std::vector<int>(_nbins);
        v = std::vector<double>(_nbins);
        v_sqd = std::vector<double>(_nbins);
    }

    HistData compute_stats() const {
        HistData data;

        data.edges.reserve(_nbins + 1);
        data.centres.reserve(_nbins);

        data.mean_v.reserve(_nbins);
        data.std_v.reserve(_nbins);

        double dz = (_domain.right_edge - _domain.left_edge) / _nbins;
        data.edges.push_back(_domain.left_edge);
        for (int i = 0; i < _nbins; i++) {
            data.edges.push_back(_domain.left_edge + dz * (i + 1));
            data.centres.push_back(_domain.left_edge + dz * (i + 0.5));

            double mean = v[i] / std::max(counts[i], 1);
            double var = v_sqd[i] / std::max(counts[i], 1) - mean * mean;

            data.mean_v.push_back(mean);
            data.std_v.push_back(std::sqrt(std::max(var, 0.0)));
        }

        data.counts = counts;
        data.nbins = _nbins;
        return data;
    }

   private:
    std::vector<int> counts;
    std::vector<double> v, v_sqd;
    Domain _domain;
    int _nbins;
};

#endif  //_HEADERS_HISTOGRAM_H_
