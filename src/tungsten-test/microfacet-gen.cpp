#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/ImageIO.hpp>
#include <io/FileUtils.hpp>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 64;

int main() {

    GaussianProcess gp(std::make_shared<HomogeneousMean>(), std::make_shared<PeriodicCovariance>(1.0f, 0.5f, TWO_PI * 5.f, Vec3f(1.0f, 1.0f, 1.f)));
    gp._covEps = 0.00001f;
    gp._maxEigenvaluesN = 1024;

    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();


    std::vector<Vec3f> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    {
        int idx = 0;
        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                points[idx] = 2.f * (Vec3f((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                derivs[idx] = Derivative::None;
                idx++;
            }
        }


        Eigen::MatrixXf samples = gp.sample(
            points.data(), derivs.data(), points.size(),
            nullptr, 0,
            Vec3f(1.0f, 0.0f, 0.0f), 1, sampler);

        std::cout << samples.minCoeff() << "-" << samples.maxCoeff() << std::endl;

        samples.array() -= samples.minCoeff();
        samples.array() /= samples.maxCoeff();
        
        ImageIO::saveHdr(incrementalFilename("microfacet-rel.exr", "", false), samples.data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
    }

}
