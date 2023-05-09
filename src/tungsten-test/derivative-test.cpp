#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 64;

int main() {

	GaussianProcess gp(std::make_shared<SphericalMean>(Vec3f(5.f, 2.5f, 0.f), 3.f), std::make_shared<SquaredExponentialCovariance>(1.0f, 1.0f));
    
    UniformPathSampler sampler(0);

    Ray ray(Vec3f(0.f), Vec3f(1.f, 0.f, 0.f), 0.0f, 10.0f);

    std::array<Vec3f, NUM_SAMPLE_POINTS+1> points;
    std::array<Derivative, NUM_SAMPLE_POINTS+1> derivs;

    std::vector<float> ts;

    for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
        float t = lerp(ray.nearT(), ray.farT(), clamp((i - sampler.next1D()) / NUM_SAMPLE_POINTS, 0.f, 1.f));
        ts.push_back(t);
        points[i] = ray.pos() + t * ray.dir();
        derivs[i] = Derivative::None;
    }

    derivs[NUM_SAMPLE_POINTS] = Derivative::None;

    {
        std::ofstream xfile("deriv-ts.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)ts.data(), sizeof(float) * ts.size());
        xfile.close();
    }

    {
        Eigen::VectorXf mean = gp.mean(points.data(), derivs.data(), ray.dir(), NUM_SAMPLE_POINTS);
        std::ofstream xfile("deriv-mean.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(float) * mean.rows() * mean.cols());
        xfile.close();
    }

    {
        Eigen::Matrix4Xf kernel(4, NUM_SAMPLE_POINTS);

        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            kernel(0, i) = (*gp._cov)(Derivative::None, Derivative::None, points[0], points[i], ray.dir());
            kernel(1, i) = (*gp._cov)(Derivative::None, Derivative::First, points[0], points[i], ray.dir());
            kernel(2, i) = (*gp._cov)(Derivative::First, Derivative::None, points[0], points[i], ray.dir());
            kernel(3, i) = (*gp._cov)(Derivative::First, Derivative::First, points[0], points[i], ray.dir());
        }

        std::ofstream xfile("kernel-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)kernel.data(), sizeof(float) * kernel.rows() * kernel.cols());
        xfile.close();
    }



    {
        std::array<Vec3f, 0> cond_pts = {  };
        std::array<Derivative, 0> cond_deriv = { };
        std::array<float, 0> cond_vs = { };
        std::array<GaussianProcess::Constraint, 1> constraints = { {0, 0, 0, FLT_MAX } };

        Eigen::MatrixXf samples = gp.sample(
            points.data(), derivs.data(), NUM_SAMPLE_POINTS,
            //cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(),
            constraints.data(), constraints.size(),
            ray.dir(), 1000, sampler);

        {
            std::ofstream xfile("deriv-samples-free.bin", std::ios::out | std::ios::binary);
            xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
            xfile.close();
        }

        std::vector<float> sampleTs;
        std::vector<Vec3f> sampleGrads;

        for (int s = 0; s < samples.cols(); s++) {
            float prevV = samples(0, s);
            float prevT = ts[0];
            for (int p = 1; p < NUM_SAMPLE_POINTS; p++) {
                float currV = samples(p, s);
                float currT = ts[p];
                if (currV < 0) {
                    float offsetT = prevV / (prevV - currV);
                    float t = lerp(prevT, currT, offsetT);
                    sampleTs.push_back(t);

                    Vec3f ip = ray.pos() + ray.dir() * t;
                    float eps = 0.01f;
                    std::array<Vec3f, 6> gradPs{
                        ip + Vec3f(eps, 0.f, 0.f),
                        ip + Vec3f(0.f, eps, 0.f),
                        ip + Vec3f(0.f, 0.f, eps),
                        ip - Vec3f(eps, 0.f, 0.f),
                        ip - Vec3f(0.f, eps, 0.f),
                        ip - Vec3f(0.f, 0.f, eps),
                    };

                    std::array<Derivative, 6> gradDerivs{
                        Derivative::None, Derivative::None, Derivative::None,
                        Derivative::None, Derivative::None, Derivative::None
                    };

                    points[NUM_SAMPLE_POINTS] = ip;

                    Eigen::VectorXf sampleValues(NUM_SAMPLE_POINTS + 1);
                    sampleValues.block(0, 0, NUM_SAMPLE_POINTS, 1) = samples.col(s);
                    sampleValues(NUM_SAMPLE_POINTS) = 0;

                    Eigen::MatrixXf gradSamples = gp.sample_cond(
                        gradPs.data(), gradDerivs.data(), gradPs.size(),
                        points.data(), sampleValues.data(), derivs.data(), points.size(),
                        nullptr, 0,
                        ray.dir(), 1, sampler);

                    Vec3f grad = Vec3f{
                        gradSamples(0,0) - gradSamples(3,0),
                        gradSamples(1,0) - gradSamples(4,0),
                        gradSamples(2,0) - gradSamples(5,0),
                    } / (2*eps);

                    sampleGrads.push_back(grad);
                    break;
                }
                prevV = currV;
                prevT = currT;
            }
        }

        Eigen::MatrixXf grads(3, sampleGrads.size());

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < sampleGrads.size(); c++) {
                grads(r, c) = sampleGrads[c][r];
            }
        }

        {
            std::ofstream xfile("deriv-dist-samples-free.bin", std::ios::out | std::ios::binary);
            xfile.write((char*)sampleTs.data(), sizeof(float) * sampleTs.size());
            xfile.close();
        }

        {
            std::ofstream xfile("deriv-grad-samples-free.bin", std::ios::out | std::ios::binary);
            xfile.write((char*)grads.data(), sizeof(float) * grads.rows() * grads.cols());
            xfile.close();
        }
    }


    /* {
        std::array<Vec3f, 2> cond_pts = { points[0], points[0] };
        std::array<Derivative, 2> cond_deriv = { Derivative::None, Derivative::First };
        std::array<float, 2> cond_vs = { 0, 1 };
        std::array<GaussianProcess::Constraint, 0> constraints = {  };

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
                    float t = lerp(prevT, currT, offsetT);
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
    }*/



    //tfile.close();


}
