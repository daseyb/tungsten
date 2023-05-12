#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/ImageIO.hpp>
#include <io/FileUtils.hpp>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 64;

size_t gidx(int i, int j) {
    return i * NUM_SAMPLE_POINTS + j;
}

Eigen::MatrixXf compute_normals(const Eigen::MatrixXf& samples) {

    Eigen::MatrixXf normals(3, NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    normals.setZero();

    auto samplesr = samples.reshaped(NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS);

    for (int i = 1; i < NUM_SAMPLE_POINTS-1; i++) {
        for (int j = 1; j < NUM_SAMPLE_POINTS-1; j++) {

            float eps = 2.f / NUM_SAMPLE_POINTS;
            
            auto r = samplesr(i + 1, j);
            auto l = samplesr(i - 1, j);
            auto b = samplesr(i, j + 1);
            auto t = samplesr(i, j - 1);

            Vec3f norm = Vec3f(-(r - l) / (2*eps), (b - t) / (2*eps), 1.f).normalized();

            normals(0, gidx(j, i)) = norm.x();
            normals(1, gidx(j, i)) = norm.y();
            normals(2, gidx(j, i)) = norm.z();
        }
    }

    return normals;
}

float compute_beckmann_roughness(const CovarianceFunction& cov) {
    float L2 = cov(Derivative::First, Derivative::First, Vec3f(0.f), Vec3f(0.f), Vec3f(1.f, 0.f, 0.f));
    return sqrt(2 * L2);
}


void normals_and_stuff(const GaussianProcess& gp) {
    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();


    std::vector<Vec3f> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    {
        {
            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    points[idx] = 2.f * (Vec3f((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }
        }


        Eigen::MatrixXf samples = gp.sample(
            points.data(), derivs.data(), points.size(),
            nullptr, 0,
            Vec3f(1.0f, 0.0f, 0.0f), 100, sampler);

        std::cout << samples.minCoeff() << "-" << samples.maxCoeff() << std::endl;


        Eigen::MatrixXf thresholded(samples.rows(), samples.cols());
        {
            for (int i = 0; i < samples.cols(); i++) {
                for (int j = 0; j < samples.rows(); j++) {
                    thresholded(j, i) = samples(j, i) < 0 ? 0 : 1;
                }
            }
        }

        for (int i = 0; i < samples.cols(); i++) {
            Eigen::MatrixXf normals = compute_normals(samples.col(i));
            normals.array() += 1.0f;
            normals.array() *= 0.5f;

            samples.col(i).array() -= samples.col(i).minCoeff();
            samples.col(i).array() /= samples.col(i).maxCoeff();

            ImageIO::saveHdr(incrementalFilename("microfacet/normals/squared-exponential-0.5/rel.exr", "", false), samples.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
            ImageIO::saveHdr(incrementalFilename("microfacet/normals/squared-exponential-0.5/thr.exr", "", false), thresholded.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
            ImageIO::saveHdr(incrementalFilename("microfacet/normals/squared-exponential-0.5/normals.exr", "", false), normals.data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 3);
        }
    }
}

void side_view(const GaussianProcess& gp) {


    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();


    std::vector<Vec3f> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    {
        {
            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    points[idx] = 2.f * (Vec3f((float)j, 0.f, (float)i) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }
        }


        Eigen::MatrixXf samples = gp.sample(
            points.data(), derivs.data(), points.size(),
            nullptr, 0,
            Vec3f(1.0f, 0.0f, 0.0f), 1, sampler);

        std::cout << samples.minCoeff() << "-" << samples.maxCoeff() << std::endl;


        Eigen::MatrixXf thresholded(samples.rows(), samples.cols());
        {
            for (int i = 0; i < samples.cols(); i++) {
                for (int j = 0; j < samples.rows(); j++) {
                    thresholded(j, i) = samples(j, i) < 0 ? 0 : 1;
                }
            }
        }

        for (int i = 0; i < samples.cols(); i++) {
            samples.col(i).array() -= samples.col(i).minCoeff();
            samples.col(i).array() /= samples.col(i).maxCoeff();

            ImageIO::saveHdr(incrementalFilename("microfacet/side-view/squared-exponential/rel.exr", "", false), samples.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
            ImageIO::saveHdr(incrementalFilename("microfacet/side-view/squared-exponential/thr.exr", "", false), thresholded.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
        }
    }
}


constexpr size_t NUM_RAY_SAMPLE_POINTS = 256;

std::vector<std::tuple<float, Vec3f>> trace_ray(const Ray& ray, const GaussianProcess* _gp, int num, PathSampleGenerator& sampler) {
    std::vector<Vec3f> points(NUM_RAY_SAMPLE_POINTS + 1);
    std::vector<Derivative> derivs(NUM_RAY_SAMPLE_POINTS + 1);
    std::vector<float> ts(NUM_RAY_SAMPLE_POINTS);

    for (int i = 0; i < NUM_RAY_SAMPLE_POINTS; i++) {
        float t = lerp(ray.nearT(), ray.farT(), clamp((i - sampler.next1D()) / NUM_RAY_SAMPLE_POINTS, 0.f, 1.f));
        ts[i] = t;
        points[i] = ray.pos() + t * ray.dir();
        derivs[i] = Derivative::None;
    }

    Eigen::MatrixXf gpSamples;

    int startSign = 1;

    std::array<Vec3f, 1> cond_pts = { points[0] };
    std::array<Derivative, 1> cond_deriv = { Derivative::None };
    std::array<float, 1> cond_vs = { _gp->sample_start_value(points[0], sampler) };
    std::array<GaussianProcess::Constraint, 1> constraints = { {0, 0, 0, FLT_MAX } };
    gpSamples = _gp->sample_cond(
        points.data(), derivs.data(), NUM_RAY_SAMPLE_POINTS,
        cond_pts.data(), cond_vs.data(), cond_deriv.data(), 0,
        constraints.data(), constraints.size(),
        ray.dir(), num, sampler);

    std::vector<std::tuple<float, Vec3f>> result;

    for (int s = 0; s < gpSamples.cols(); s++) {
        float prevV = gpSamples(0, 0);
        float prevT = ts[0];
        for (int p = 1; p < NUM_RAY_SAMPLE_POINTS; p++) {
            float currV = gpSamples(p, 0);
            float currT = ts[p];
            if (currV < 0) {
                float offsetT = prevV / (prevV - currV);
                float t = lerp(prevT, currT, offsetT);

                Vec3f ip = ray.pos() + ray.dir() * t;
                float eps = 0.001f;
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

                points[NUM_RAY_SAMPLE_POINTS] = ip;

                Eigen::VectorXf sampleValues(NUM_RAY_SAMPLE_POINTS + 1);
                sampleValues.block(0, 0, NUM_RAY_SAMPLE_POINTS, 1) = gpSamples.col(0);
                sampleValues(NUM_RAY_SAMPLE_POINTS) = 0;

                Eigen::MatrixXf gradSamples = _gp->sample_cond(
                    gradPs.data(), gradDerivs.data(), gradPs.size(),
                    points.data(), sampleValues.data(), derivs.data(), points.size(),
                    nullptr, 0,
                    ray.dir(), 1, sampler);

                Vec3f grad = Vec3f{
                    gradSamples(0,0) - gradSamples(3,0),
                    gradSamples(1,0) - gradSamples(4,0),
                    gradSamples(2,0) - gradSamples(5,0),
                } / (2 * eps) * startSign;

                if (!std::isfinite(grad.avg())) {
                    std::cout << "Gradient invalid.\n";
                }
                else {
                    result.emplace_back(t, grad);
                }

                break;
            }
            prevV = currV;
            prevT = currT;
        }
    }

    return result;
}

void v_ndf(const GaussianProcess& gp) {
    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();

    float angle = 0;// PI * 0.25;

    Ray ray = Ray(Vec3f(0.f, 0.f, 50.f), Vec3f(sin(angle), 0.f, -cos(angle)));

    ray.setNearT(-(ray.pos().z()-10.0f) / ray.dir().z());
    ray.setFarT(-(ray.pos().z()+10.0f) / ray.dir().z());

    auto trace_results = trace_ray(ray, &gp, 10000, sampler);

    Eigen::MatrixXf normals(trace_results.size(),3);

    for (int i = 0; i < trace_results.size(); i++) {
        auto [t, n] = trace_results[i];
        n.normalize();
        normals(i,0) = n[0];
        normals(i,1) = n[1];
        normals(i,2) = n[2];
    }

    {
        std::ofstream xfile(incrementalFilename("microfacet/visible-normals/squared-exponential.bin", "", false).asString(), std::ios::out | std::ios::binary);
        xfile.write((char*)normals.data(), sizeof(float) * normals.rows() * normals.cols());
        xfile.close();
    }

}

int main() {
    GaussianProcess gp(std::make_shared<LinearMean>(Vec3f(0.f), Vec3f(0.f, 0.f, 1.f), 1.0f), std::make_shared<SquaredExponentialCovariance>(1.0f, 0.5f, Vec3f(1.0f, 1.0f, 0.0001f)));
    gp._covEps = 0.00001f;
    gp._maxEigenvaluesN = 1024;

    std::cout << "Beckmann roughness: " << compute_beckmann_roughness(*gp._cov) << "\n";

    normals_and_stuff(gp);
    //side_view(gp);
    //v_ndf(gp);
}
