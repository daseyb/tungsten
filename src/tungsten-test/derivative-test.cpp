#include <core/math/GaussianProcess.hpp>
#include <core/media/GaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 128;

int main() {

	auto gp = std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<SquaredExponentialCovariance>(1.0f, 1.0f));

    GaussianProcessMedium gp_med(gp, 0, 1, 1, NUM_SAMPLE_POINTS);
    gp_med.prepareForRender();
    
    UniformPathSampler sampler(0);
    sampler.next2D();

    Ray ray(Vec3f(0.f), Vec3f(1.f, 0.f, 0.f), 0.0f, 15.0f);
    
    std::vector<float> ts;

    float L0 = (*gp->_cov)(Derivative::None, Derivative::None, Vec3f(0.f), Vec3f(0.f), Vec3f(-1.f, 0.f, 0.f), Vec3f(1.f, 0.f, 0.f));
    float L2 = (*gp->_cov)(Derivative::First, Derivative::First, Vec3f(0.f), Vec3f(0.f), Vec3f(-1.f, 0.f, 0.f), Vec3f(1.f, 0.f, 0.f));

    std::cout << "L0:" << L0 << std::endl;
    std::cout << "L2:" << L2 << std::endl;

    int sample_count = 10000;
    std::cout << "fp" << std::endl;
    {
        std::vector<float> ts;
        for (int i = 0; i < sample_count; i++) {
            if (i % 100 == 0) {
                std::cout << i << "/" << sample_count << "\r";
            }

            Medium::MediumState state;
            state.reset();
            MediumSample sample;
            if (gp_med.sampleDistance(sampler, ray, state, sample)) {
                ts.push_back(sample.continuedT);
            }
        }

        {
            std::ofstream xfile(tinyformat::format("transmittance/%s-sample-fp.bin", gp->_cov->id()), std::ios::out | std::ios::binary);
            xfile.write((char*)ts.data(), sizeof(float) * ts.size());
            xfile.close();
        }
    }


    std::cout << "\npp" << std::endl;
    {
        std::vector<float> ts;
        for (int i = 0; i < sample_count; i++) {
            if (i % 100 == 0) {
                std::cout << i << "/" << sample_count << "\r";
            }

            Medium::MediumState state;
            state.reset();
            state.firstScatter = false;
            MediumSample sample;
            if (gp_med.sampleDistance(sampler, ray, state, sample)) {
                ts.push_back(sample.continuedT);
            }
        }

        {
            std::ofstream xfile(tinyformat::format("transmittance/%s-sample-pp.bin", gp->_cov->id()), std::ios::out | std::ios::binary);
            xfile.write((char*)ts.data(), sizeof(float) * ts.size());
            xfile.close();
        }
    }

}
