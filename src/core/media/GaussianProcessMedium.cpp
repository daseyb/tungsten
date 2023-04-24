#include "GaussianProcessMedium.hpp"

#include "sampling/PathSampleGenerator.hpp"

#include "math/GaussianProcess.hpp"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

namespace Tungsten {

    constexpr size_t NUM_SAMPLE_POINTS = 16;

    GaussianProcessMedium::GaussianProcessMedium()
: _materialSigmaA(0.0f),
  _materialSigmaS(0.0f),
  _density(1.0f),
  _gp(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(), std::make_shared<SquaredExponentialCovariance>()))
{
}

void GaussianProcessMedium::fromJson(JsonPtr value, const Scene &scene)
{
    Medium::fromJson(value, scene);
    value.getField("sigma_a", _materialSigmaA);
    value.getField("sigma_s", _materialSigmaS);
    value.getField("density", _density);

    if (auto gp = value["gaussian_process"])
        _gp = scene.fetchGaussianProcess(gp);

}

rapidjson::Value GaussianProcessMedium::toJson(Allocator &allocator) const
{
    return JsonObject{Medium::toJson(allocator), allocator,
        "type", "gaussian_process",
        "sigma_a", _materialSigmaA,
        "sigma_s", _materialSigmaS,
        "density", _density,
        "gaussian_process", *_gp
    };
}

bool GaussianProcessMedium::isHomogeneous() const
{
    return false;
}

void GaussianProcessMedium::prepareForRender()
{
    _sigmaA = _materialSigmaA*_density;
    _sigmaS = _materialSigmaS*_density;
    _sigmaT = _sigmaA + _sigmaS;
    _absorptionOnly = _sigmaS == 0.0f;
}

Vec3f GaussianProcessMedium::sigmaA(Vec3f /*p*/) const
{
    return _sigmaA;
}

Vec3f GaussianProcessMedium::sigmaS(Vec3f /*p*/) const
{
    return _sigmaS;
}

Vec3f GaussianProcessMedium::sigmaT(Vec3f /*p*/) const
{
    return _sigmaT;
}

bool GaussianProcessMedium::sampleDistance(PathSampleGenerator &sampler, const Ray &ray,
        MediumState &state, MediumSample &sample) const
{
    sample.emission = Vec3f(0.0f);

    if (state.bounce > _maxBounce)
        return false;

    float maxT = ray.farT();
    if (_absorptionOnly) {
        if (maxT == Ray::infinity())
            return false;
        sample.t = maxT;
        sample.weight = transmittance(sampler, ray, state.firstScatter, true);
        sample.pdf = 1.0f;
        sample.exited = true;
    } else {
        int component = sampler.nextDiscrete(3);
        float sigmaTc = _sigmaT[component];

        float t = maxT;
        {
            std::array<Vec3f, NUM_SAMPLE_POINTS*2> points;
            std::array<Derivative, NUM_SAMPLE_POINTS*2> derivs;

            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                float t = lerp(ray.nearT(), ray.farT(), (i + sampler.next1D()) / NUM_SAMPLE_POINTS);
                points[i*2] = points[i] = ray.pos() + t * ray.dir();
                derivs[i] = Derivative::None;
                derivs[i*2] = Derivative::First;
            }

            Eigen::MatrixXf gpSamples;

            if (state.firstScatter) {
                std::array<GaussianProcess::Constraint, 1> constraints = { {0, 0, 0, FLT_MAX } };
                gpSamples = _gp->sample(
                    points.data(), derivs.data(), points.size(),
                    constraints.data(), constraints.size(),
                    ray.dir(), 1, sampler);
            }
            else {
                std::array<Vec3f, 1> cond_pts = { points[0] };
                std::array<Derivative, 1> cond_deriv = { Derivative::None };
                std::array<float, 1> cond_vs = { 0 };
                std::array<GaussianProcess::Constraint, 1> constraints = { {NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 0, FLT_MAX } };

                gpSamples = _gp->sample_cond(
                    points.data(), derivs.data(), points.size(),
                    cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(),
                    constraints.data(), constraints.size(),
                    ray.dir(), 1, sampler);
            }
            

            float prevV = gpSamples(0, 0);
            for (int p = 0; p < gpSamples.rows(); p++) {
                float currV = gpSamples(p, 0);
                if (currV <= 0) {
                    float offsetT = prevV / (prevV - currV);
                    t = lerp(ray.nearT(), ray.farT(), float(p - 1 + offsetT) / points.size());
                    //sample.aniso = gpSamples(p * 2, 0);
                    break;
                }
                prevV = currV;
            }
        }

        sample.t = min(t, maxT);
        sample.continuedT = t;
        sample.exited = (t >= maxT);
        sample.weight = Vec3f(1.);
        sample.pdf = 1;
        
        state.advance();
    }
    sample.p = ray.pos() + sample.t*ray.dir();
    sample.phase = _phaseFunction.get();

    return true;
}

Vec3f GaussianProcessMedium::transmittance(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface,
        bool endOnSurface) const
{
    if (ray.farT() == Ray::infinity())
        return Vec3f(0.0f);

    std::array<Vec3f, NUM_SAMPLE_POINTS> points;
    std::array<Derivative, NUM_SAMPLE_POINTS> derivs;

    for (int i = 0; i < points.size(); i++) {
        float t = lerp(ray.nearT(), ray.farT(),  (i + sampler.next1D()) / NUM_SAMPLE_POINTS);
        points[i] = ray.pos() + t * ray.dir();
        derivs[i] = Derivative::None;
    }


    Eigen::MatrixXf gpSamples;

    if (startOnSurface) {
        std::array<GaussianProcess::Constraint, 1> constraints = { {0, 0, 0, FLT_MAX } };
        gpSamples = _gp->sample(
            points.data(), derivs.data(), points.size(),
            constraints.data(), constraints.size(),
            ray.dir(), 10, sampler);
    }
    else {
        std::array<Vec3f, 1> cond_pts = { points[0] };
        std::array<Derivative, 1> cond_deriv = { Derivative::None };
        std::array<float, 1> cond_vs = { 0 };
        //std::array<GaussianProcess::Constraint, 1> constraints = { {NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 0, FLT_MAX } };
        std::array<GaussianProcess::Constraint, 0> constraints = { };

        gpSamples = _gp->sample_cond(
            points.data(), derivs.data(), points.size(),
            cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(),
            constraints.data(), constraints.size(),
            ray.dir(), 10, sampler);
    }

    int madeItCnt = 0;
    for (int s = 0; s < gpSamples.cols(); s++) {
        
        bool madeIt = true;
        for (int p = 0; p < gpSamples.rows(); p++) {
            if (gpSamples(p, s) <= 0) {
                madeIt = false;
                break;
            }
        }

        if (madeIt) madeItCnt++;
    }

    return Vec3f(float(madeItCnt) / gpSamples.cols());
}

float GaussianProcessMedium::pdf(PathSampleGenerator &/*sampler*/, const Ray &ray, bool startOnSurface, bool endOnSurface) const
{
    return 1.0f;
}

}
