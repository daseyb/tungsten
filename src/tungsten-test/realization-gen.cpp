#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 128;

int main() {

	GaussianProcess gp(std::make_shared<SphericalMean>(Vec3f(0.f, 0.f, 0.f), 0.25f), std::make_shared<SquaredExponentialCovariance>(0.1f, 0.05f));

    
    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();


    std::vector<Vec3f> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    {
        int idx = 0;
        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    points[idx] = 2.f * (Vec3f((float)i, (float)j, (float)k) / (NUM_SAMPLE_POINTS-1) - 0.5f);
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }
        }

        Eigen::MatrixXf samples = gp.sample(
            points.data(), derivs.data(), points.size(),
            nullptr, 0,
            Vec3f(1.0f, 0.0f, 0.0f), 1, sampler);

        {
            std::ofstream xfile("grid-samples.bin", std::ios::out | std::ios::binary);
            xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
            xfile.close();
        }
    }
   
}
