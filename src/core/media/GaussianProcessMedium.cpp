#include "GaussianProcessMedium.hpp"

#include <cfloat>

#include "sampling/PathSampleGenerator.hpp"

#include "math/GaussianProcess.hpp"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

namespace Tungsten {

std::string GaussianProcessMedium::correlationContextToString(GPCorrelationContext ctxt)
{
    switch (ctxt) {
    default:
    case GPCorrelationContext::Elephant:  return "elephant";
    case GPCorrelationContext::Goldfish:   return "goldfish";
    case GPCorrelationContext::Dori:   return "dori";
    }
}

GPCorrelationContext GaussianProcessMedium::stringToCorrelationContext(const std::string& name)
{
    if (name == "elephant")
        return GPCorrelationContext::Elephant;
    else if (name == "goldfish")
        return GPCorrelationContext::Goldfish;
    else if (name == "dori")
        return GPCorrelationContext::Dori;
    FAIL("Invalid correaltion context: '%s'", name);
}


    GaussianProcessMedium::GaussianProcessMedium()
: _materialSigmaA(0.0f),
  _materialSigmaS(0.0f),
  _density(1.0f),
  _gp(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(), std::make_shared<SquaredExponentialCovariance>())),
  _samplePoints(32),
  _ctxt(GPCorrelationContext::Goldfish)
{
}

void GaussianProcessMedium::fromJson(JsonPtr value, const Scene &scene)
{
    Medium::fromJson(value, scene);
    value.getField("sigma_a", _materialSigmaA);
    value.getField("sigma_s", _materialSigmaS);
    value.getField("density", _density);
    value.getField("sample_points", _samplePoints);

    std::string ctxtString = "goldfish";
    value.getField("correlation_context", ctxtString);
    _ctxt = stringToCorrelationContext(ctxtString);

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
        "sample_points", _samplePoints,
        "gaussian_process", *_gp,
        "correlation_context", correlationContextToString(_ctxt)
    };
}

void GaussianProcessMedium::loadResources() {
    _gp->loadResources();
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
        sample.weight = transmittance(sampler, ray, state.firstScatter, true, &sample);
        sample.pdf = 1.0f;
        sample.exited = true;
    } else {
        float t = maxT;
        {
            std::vector<Vec3f> points(_samplePoints+1);
            std::vector<Derivative> derivs(_samplePoints+1);
            std::vector<float> ts(_samplePoints);

            float tOffset = sampler.next1D();

            for (int i = 0; i < _samplePoints; i++) {
                float t = lerp(ray.nearT(), min(100.f, ray.farT()), clamp((i - tOffset) / _samplePoints, 0.f, 1.f));
                ts[i] = t;
                points[i] = ray.pos() + t * ray.dir();
                derivs[i] = Derivative::None;
            }

            Eigen::MatrixXd gpSamples;

            int startSign = 1;

            if (state.firstScatter) {
                std::array<Vec3f, 1> cond_pts = { points[0] };
                std::array<Derivative, 1> cond_deriv = { Derivative::None };
                std::array<double, 1> cond_vs = { _gp->sample_start_value(points[0], sampler) };
                std::array<GaussianProcess::Constraint, 1> constraints = { {0, 0, 0, FLT_MAX } };
                gpSamples = _gp->sample_cond(
                    points.data(), derivs.data(), _samplePoints, nullptr,
                    cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
                    constraints.data(), constraints.size(),
                    ray.dir(), 1, sampler);
            }
            else {

                std::array<Vec3f, 2> cond_pts = { points[0], points[0] };
                std::array<Derivative, 2> cond_deriv = { Derivative::None, Derivative::First };

                float deriv = state.lastAniso.dot(ray.dir().normalized());
                startSign = deriv < 0 ? -1 : 1;
                std::array<double, 2> cond_vs = { 0, deriv };

                gpSamples = _gp->sample_cond(
                    points.data(), derivs.data(), _samplePoints, nullptr,
                    cond_pts.data(), cond_vs.data(), cond_deriv.data(), _ctxt == GPCorrelationContext::Dori ? cond_pts.size() : 1, nullptr,
                    nullptr, 0,
                    ray.dir(), 1, sampler) * startSign;
            }
            

            float prevV = gpSamples(0, 0);

            if (state.firstScatter && prevV < 0) {
                return false;
            }

            float prevT = ts[0];
            for (int p = 1; p < _samplePoints; p++) {
                float currV = gpSamples(p, 0);
                float currT = ts[p];
                if (currV < 0) {
                    float offsetT = prevV / (prevV - currV);
                    t = lerp(prevT, currT, offsetT);

                    Vec3f ip = ray.pos() + ray.dir() * t;

#if 0
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

                    //points[p] = ip;
                    //gpSamples(p, 0) = 0;

                    //Eigen::VectorXf sampleValues(_samplePoints+1);
                    //sampleValues.block(0, 0, _samplePoints, 1) = gpSamples.col(0);
                    //sampleValues(_samplePoints) = 0;

                    Eigen::MatrixXf fgpSamples = gpSamples.cast<float>();

                    Eigen::MatrixXd gradSamples = _gp->sample_cond(
                        gradPs.data(), gradDerivs.data(), gradPs.size(), nullptr,
                        points.data(), fgpSamples.data(), derivs.data(), p, nullptr,
                        nullptr, 0,
                        ray.dir(), 1, sampler);

                    Vec3d gradd = Vec3d{
                        gradSamples(0,0) - gradSamples(3,0),
                        gradSamples(1,0) - gradSamples(4,0),
                        gradSamples(2,0) - gradSamples(5,0),
                    } / (2 * eps);

                    Vec3f grad = Vec3f((float)gradd.x(), (float)gradd.y(), (float)gradd.z());

#elif 1
                    std::array<Vec3f, 1> gradPs{ ip };
                    std::array<Derivative, 1> gradDerivs{ Derivative::First };

                    points[_samplePoints] = ip;
                    derivs[_samplePoints] = Derivative::None;

                    Eigen::VectorXd sampleValues(_samplePoints + 1);
                    sampleValues.block(0, 0, _samplePoints, 1) = gpSamples.col(0);
                    sampleValues(_samplePoints) = 0;

                    Vec3f grad;

                    grad.x() = _gp->sample_cond(
                        gradPs.data(), gradDerivs.data(), gradPs.size(), nullptr,
                        points.data(), sampleValues.data(), derivs.data(), _samplePoints + 1, nullptr,
                        nullptr, 0,
                        Vec3f(1.f, 0.f, 0.f), 1, sampler)(0, 0);

                    grad.y() = _gp->sample_cond(
                        gradPs.data(), gradDerivs.data(), gradPs.size(), nullptr,
                        points.data(), sampleValues.data(), derivs.data(), _samplePoints + 1, nullptr,
                        nullptr, 0,
                        Vec3f(0.f, 1.f, 0.f), 1, sampler)(0, 0);

                    grad.z() = _gp->sample_cond(
                        gradPs.data(), gradDerivs.data(), gradPs.size(), nullptr,
                        points.data(), sampleValues.data(), derivs.data(), _samplePoints + 1, nullptr,
                        nullptr, 0,
                        Vec3f(0.f, 0.f, 1.f), 1, sampler)(0, 0);
#else
                    std::array<Vec3f, 3> gradPs{ ip, ip, ip };
                    std::array<Derivative, 3> gradDerivs{ Derivative::First, Derivative::First, Derivative::First };
                    std::array<Vec3f, 3> gradDirs{
                        Vec3f(1.f, 0.f, 0.f),
                        Vec3f(0.f, 1.f, 0.f),
                        Vec3f(0.f, 0.f, 1.f),
                    };

                    points[_samplePoints] = ip;
                    derivs[_samplePoints] = Derivative::None;

                    Eigen::VectorXf sampleValues(_samplePoints + 1);
                    sampleValues.block(0, 0, _samplePoints, 1) = gpSamples.col(0).cast<float>();
                    sampleValues(_samplePoints) = 0;

                    auto gradSamples = _gp->sample_cond(
                        gradPs.data(), gradDerivs.data(), gradPs.size(), gradDirs.data(),
                        points.data(), sampleValues.data(), derivs.data(), _samplePoints+1, nullptr,
                        nullptr, 0,
                        ray.dir(), 1, sampler);

                    Vec3f grad = {
                        (float)gradSamples(0,0), (float)gradSamples(1,0), (float)gradSamples(2,0)
                    };
#endif

                    if (grad.dot(ray.dir()) > 0) {
                        return false;
                    }

                    sample.aniso = grad * startSign;
                    if (!std::isfinite(sample.aniso.avg())) {
                        sample.aniso = Vec3f(1.f, 0.f, 0.f);
                        std::cout << "Gradient invalid.\n";
                        return false;
                    }

                    if (sample.aniso.lengthSq() < 0.0000001f) {
                        sample.aniso = Vec3f(1.f, 0.f, 0.f);
                        std::cout << "Gradient zero.\n";
                        return false;
                    }
                    break;
                }
                prevV = currV;
                prevT = currT;
            }
        }

        
        sample.t = min(t, maxT);
        sample.continuedT = t;
        sample.exited = (t >= maxT);
        sample.weight = sigmaS(ray.pos() + sample.t * ray.dir()) / sigmaT(ray.pos() + sample.t * ray.dir());
        sample.continuedWeight = sigmaS(ray.pos() + sample.continuedT * ray.dir()) / sigmaT(ray.pos() + sample.continuedT * ray.dir());
        sample.pdf = 1;
        
        state.lastAniso = sample.aniso;
        state.advance();
    }
    sample.p = ray.pos() + sample.t*ray.dir();
    sample.phase = _phaseFunction.get();

    return true;
}

Vec3f GaussianProcessMedium::transmittance(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface,
        bool endOnSurface, MediumSample* sample) const
{
    if (ray.farT() == Ray::infinity())
        return Vec3f(0.0f);

    std::vector<Vec3f> points(_samplePoints);
    std::vector<Derivative> derivs(_samplePoints);

    for (int i = 0; i < points.size(); i++) {
        float t = lerp(ray.nearT(), ray.farT(),  (i + sampler.next1D()) / _samplePoints);
        points[i] = ray.pos() + t * ray.dir();
        derivs[i] = Derivative::None;
    }

    Eigen::MatrixXd gpSamples;
    int startSign = 1;

    if (startOnSurface) {
        std::array<Vec3f, 1> cond_pts = { points[0] };
        std::array<Derivative, 1> cond_deriv = { Derivative::None };
        std::array<double, 1> cond_vs = { _gp->sample_start_value(points[0], sampler) };
        std::array<GaussianProcess::Constraint, 1> constraints = { {0, 0, 0, FLT_MAX } };
        gpSamples = _gp->sample_cond(
            points.data(), derivs.data(), points.size(), nullptr,
            cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
            constraints.data(), constraints.size(),
            ray.dir(), 10, sampler);
    }
    else {

        if (!sample) {
            std::cout << "what\n";
            return Vec3f(0.f);
        }

        std::array<Vec3f, 2> cond_pts = { points[0], points[0] };
        std::array<Derivative, 2> cond_deriv = { Derivative::None, Derivative::First };

        float deriv = sample->aniso.dot(ray.dir().normalized());
        startSign = deriv < 0 ? -1 : 1;
        std::array<double, 2> cond_vs = { 0, deriv };

        gpSamples = _gp->sample_cond(
            points.data(), derivs.data(), points.size(), nullptr,
            cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
            nullptr, 0,
            ray.dir(), 10, sampler) * startSign;
    }

    int madeItCnt = 0;
    for (int s = 0; s < gpSamples.cols(); s++) {
        
        bool madeIt = true;
        for (int p = 0; p < _samplePoints; p++) {
            if (gpSamples(p, s) < 0) {
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
