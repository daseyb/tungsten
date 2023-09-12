#include <core/media/GaussianProcessMedium.hpp>
#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <tinyformat/tinyformat.hpp>

using namespace Tungsten;

void compute_spectral_density(std::shared_ptr<GaussianProcess> gp) {
    gp->loadResources();

    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    {
        std::vector<double> spectralDensity;
        size_t num_samples = 1000;
        double max_w = 10;
        for (size_t i = 0; i < num_samples; i++) {
            spectralDensity.push_back(gp->_cov->spectral_density(double(i) / num_samples * max_w));
        }

        {
            std::ofstream xfile(
                (basePath / Path("spectral_density.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)spectralDensity.data(), sizeof(double) * spectralDensity.size());
            xfile.close();
        }
    }


    {
        UniformPathSampler sampler(0);
        sampler.next2D();

        std::vector<double> spectralDensitySamples;
        size_t num_samples = 100000;
        for (size_t i = 0; i < num_samples; i++) {
            spectralDensitySamples.push_back(gp->_cov->sample_spectral_density(sampler));
        }

        {
            std::ofstream xfile(
                (basePath / Path("spectral_density_samples.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)spectralDensitySamples.data(), sizeof(double) * spectralDensitySamples.size());
            xfile.close();
        }
    }
}

int main() {
    compute_spectral_density(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<SquaredExponentialCovariance>(1.f, 1.f)));
    compute_spectral_density(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 1.f, 1.0f)));
    compute_spectral_density(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 1.f, 0.5f)));
    compute_spectral_density(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 1.f, 5.0f)));
}
