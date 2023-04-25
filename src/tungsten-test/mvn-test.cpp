#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 64;

int main() {

	GaussianProcess gp(std::make_shared<HomogeneousMean>(), std::make_shared<SquaredExponentialCovariance>());

    UniformPathSampler sampler(0);

    Ray ray(Vec3f(0.f), Vec3f(1.f, 0.f, 0.f), 0.0f, 10.0f);

    std::array<Vec3f, NUM_SAMPLE_POINTS + 1> points;
    std::array<Derivative, NUM_SAMPLE_POINTS + 1> derivs;

    std::vector<float> ts;

    for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
        float t = lerp(ray.nearT(), ray.farT(), clamp((i - sampler.next1D()) / NUM_SAMPLE_POINTS, 0.f, 1.f));
        ts.push_back(t);
        points[i] = ray.pos() + t * ray.dir();
        derivs[i] = Derivative::None;
    }

    points[NUM_SAMPLE_POINTS] = points[0];
    derivs[NUM_SAMPLE_POINTS] = Derivative::First;


    {
        std::vector<float> normalSamples;
        // Box muller transform
        for (int i = 0; i < 1000; i++) {
            Vec2f samples = gp.rand_normal_2(sampler);
            normalSamples.push_back(samples.x());
            normalSamples.push_back(samples.y());
        }

        std::ofstream xfile("normalSamples.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)normalSamples.data(), sizeof(float) * normalSamples.size());
        xfile.close();
    }
    

    {
        std::ofstream xfile("ts.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)ts.data(), sizeof(float) * ts.size());
        xfile.close();
    }

    {
        Eigen::VectorXf mean = gp.mean(points.data(), derivs.data(), ray.dir(), points.size());
        std::ofstream xfile("mean.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(float) * mean.rows() * mean.cols());
        xfile.close();
    }

    {
        Eigen::MatrixXf cov = gp.cov(points.data(), points.data(), derivs.data(), derivs.data(), points.size(), points.size());
        std::ofstream xfile("cov.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)cov.data(), sizeof(float) * cov.rows() * cov.cols());
        xfile.close();
    }

    {
        Eigen::MatrixXf samples = gp.sample(points.data(), derivs.data(), points.size(), nullptr, 0, ray.dir(), 50, sampler);
        std::ofstream xfile("samples.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
        xfile.close();
    }

    {
        Eigen::MatrixXf samples = gp.sample(points.data(), derivs.data(), points.size() - 1, nullptr, 0, ray.dir(), 50, sampler);
        std::ofstream xfile("samples-nod.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
        xfile.close();
    }

    {
        std::array<Vec3f, 1> cond_pts = { points[0] };
        std::array<Derivative, 1> cond_deriv = { Derivative::None };
        std::array<float, 1> cond_vs = { 0 };

        Eigen::MatrixXf samples = gp.sample_cond(
            points.data(), derivs.data(), points.size(),
            cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(),
            nullptr, 0,
            ray.dir(), 50, sampler);

        std::ofstream xfile("samples-cond.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
        xfile.close();
    }

    {
        std::array<Vec3f, 1> cond_pts = { points[0] };
        std::array<Derivative, 1> cond_deriv = { Derivative::None };
        std::array<float, 1> cond_vs = { 0 };
        std::array<GaussianProcess::Constraint, 1> constraints = { {NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 0, FLT_MAX } };

        Eigen::MatrixXf samples = gp.sample_cond(
            points.data(), derivs.data(), points.size(),
            cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(),
            constraints.data(), constraints.size(),
            ray.dir(), 50, sampler);

        std::ofstream xfile("samples-cond-const.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
        xfile.close();
    }

    {
        std::array<Vec3f, 1> cond_pts = { points[0] };
        std::array<Derivative, 1> cond_deriv = { Derivative::None };
        std::array<float, 1> cond_vs = { 0 };
        std::array<GaussianProcess::Constraint, 1> constraints = { {NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 0, FLT_MAX } };

        Eigen::MatrixXf samples = gp.sample_cond(
            points.data(), derivs.data(), points.size(),
            cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(),
            constraints.data(), constraints.size(),
            ray.dir(), 50000, sampler);

        std::vector<float> sampleTs;
        for (int s = 0; s < samples.cols(); s++) {
            float prevV = samples(0, s);
            float prevT = ts[0];
            for (int p = 1; p < NUM_SAMPLE_POINTS; p++) {
                float currV = samples(p, s);
                float currT = ts[p];
                if (currV < 0) {
                    float offsetT = prevV / (prevV - currV);
                    float t = currT; // lerp(prevT, currT, offsetT);
                    sampleTs.push_back(t);
                    //sample.aniso = gpSamples(p * 2, 0);
                    break;
                }
                prevV = currV;
                prevT = currT;
            }
        }

        std::ofstream xfile("dist-samples-cond.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)sampleTs.data(), sizeof(float) * sampleTs.size());
        xfile.close();
    }



    //tfile.close();


}