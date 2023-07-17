#include "GaussianProcessMedium.hpp"

#include <cfloat>

#include "sampling/PathSampleGenerator.hpp"

#include "math/GaussianProcess.hpp"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "bsdfs/Microfacet.hpp"

//#define NORMALS_FD
//#define NORMALS_NC
#define NORMALS_WC

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
        FAIL("Invalid correlation context: '%s'", name);
    }

    std::string GaussianProcessMedium::intersectMethodToString(GPIntersectMethod ctxt)
    {
        switch (ctxt) {
        default:
        case GPIntersectMethod::Mean:  return "mean";
        case GPIntersectMethod::MeanRaymarch:  return "mean_raymarch";
        case GPIntersectMethod::GPDiscrete:   return "gp_discrete";
        }
    }

    GPIntersectMethod GaussianProcessMedium::stringToIntersectMethod(const std::string& name)
    {
        if (name == "mean")
            return GPIntersectMethod::Mean;
        else if (name == "mean_raymarch")
            return GPIntersectMethod::MeanRaymarch;
        else if (name == "gp_discrete")
            return GPIntersectMethod::GPDiscrete;
        FAIL("Invalid intersect method: '%s'", name);
    }

    std::string GaussianProcessMedium::normalSamplingMethodToString(GPNormalSamplingMethod val)
    {
        switch (val) {
        default:
        case GPNormalSamplingMethod::FiniteDifferences:  return "finite_differences";
        case GPNormalSamplingMethod::ConditionedGaussian:   return "conditioned_gaussian";
        case GPNormalSamplingMethod::Beckmann:   return "beckmann";
        }
    }

    GPNormalSamplingMethod GaussianProcessMedium::stringToNormalSamplingMethod(const std::string& name)
    {
        if (name == "finite_differences")
            return GPNormalSamplingMethod::FiniteDifferences;
        else if (name == "conditioned_gaussian")
            return GPNormalSamplingMethod::ConditionedGaussian;
        else if (name == "beckmann")
            return GPNormalSamplingMethod::Beckmann;
        FAIL("Invalid normal sampling method: '%s'", name);
    }


    GaussianProcessMedium::GaussianProcessMedium()
        : _materialSigmaA(0.0f),
        _materialSigmaS(0.0f),
        _density(1.0f),
        _gp(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(), std::make_shared<SquaredExponentialCovariance>())),
        _samplePoints(32),
        _ctxt(GPCorrelationContext::Goldfish),
        _intersectMethod(GPIntersectMethod::GPDiscrete),
        _normalSamplingMethod(GPNormalSamplingMethod::ConditionedGaussian)
    {
    }

    void GaussianProcessMedium::fromJson(JsonPtr value, const Scene& scene)
    {
        Medium::fromJson(value, scene);
        value.getField("sigma_a", _materialSigmaA);
        value.getField("sigma_s", _materialSigmaS);
        value.getField("density", _density);
        value.getField("sample_points", _samplePoints);

        std::string ctxtString = "goldfish";
        value.getField("correlation_context", ctxtString);
        _ctxt = stringToCorrelationContext(ctxtString);

        std::string intersectString = "gp_discrete";
        value.getField("intersect_method", intersectString);
        _intersectMethod = stringToIntersectMethod(intersectString);

        std::string normalString = "conditioned_gaussian";
        value.getField("normal_method", normalString);
        _normalSamplingMethod = stringToNormalSamplingMethod(normalString);

        if (auto gp = value["gaussian_process"])
            _gp = scene.fetchGaussianProcess(gp);

    }

    rapidjson::Value GaussianProcessMedium::toJson(Allocator& allocator) const
    {
        return JsonObject{ Medium::toJson(allocator), allocator,
            "type", "gaussian_process",
            "sigma_a", _materialSigmaA,
            "sigma_s", _materialSigmaS,
            "density", _density,
            "sample_points", _samplePoints,
            "gaussian_process", *_gp,
            "correlation_context", correlationContextToString(_ctxt),
            "intersect_method", intersectMethodToString(_intersectMethod),
            "normal_method", normalSamplingMethodToString(_normalSamplingMethod),
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
        _sigmaA = _materialSigmaA * _density;
        _sigmaS = _materialSigmaS * _density;
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

    bool GaussianProcessMedium::sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3f& ip,
        MediumState& state, Vec3f& grad) const {

        GPContextFunctionSpace& ctxt = *(GPContextFunctionSpace*)state.gpContext.get();

        switch(_normalSamplingMethod) {
            case GPNormalSamplingMethod::FiniteDifferences:
            {
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

                Eigen::MatrixXd gradSamples = _gp->sample_cond(
                    gradPs.data(), gradDerivs.data(), gradPs.size(), nullptr,
                    ctxt.points.data(), ctxt.values.data(), ctxt.derivs.data(), ctxt.points.size(), nullptr,
                    nullptr, 0,
                    ray.dir(), 1, sampler);

                Vec3d gradd = Vec3d{
                    gradSamples(0,0) - gradSamples(3,0),
                    gradSamples(1,0) - gradSamples(4,0),
                    gradSamples(2,0) - gradSamples(5,0),
                } / (2 * eps);

                grad = Vec3f((float)gradd.x(), (float)gradd.y(), (float)gradd.z());
                break;
            }
            case GPNormalSamplingMethod::ConditionedGaussian:
            {
                std::array<Vec3f, 3> gradPs{ ip, ip, ip };
                std::array<Derivative, 3> gradDerivs{ Derivative::First, Derivative::First, Derivative::First };
                std::array<Vec3f, 3> gradDirs{
                    Vec3f(1.f, 0.f, 0.f),
                    Vec3f(0.f, 1.f, 0.f),
                    Vec3f(0.f, 0.f, 1.f),
                };

                auto gradSamples = _gp->sample_cond(
                    gradPs.data(), gradDerivs.data(), gradPs.size(), gradDirs.data(),
                    ctxt.points.data(), ctxt.values.data(), ctxt.derivs.data(), ctxt.points.size(), nullptr,
                    nullptr, 0,
                    ray.dir(), 1, sampler);

                grad = {
                    (float)gradSamples(0,0), (float)gradSamples(1,0), (float)gradSamples(2,0)
                };
                break;
            }
            case GPNormalSamplingMethod::Beckmann:
            {
                static Microfacet::Distribution distribution("beckmann");
                grad = Microfacet::sample(distribution, _gp->_cov->compute_beckmann_roughness(), sampler.next2D());
                std::swap(grad.y(), grad.z());
                break;
            }
        }

        //grad *= state.startSign;

        return true;
    }

    bool GaussianProcessMedium::intersect(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, float& t) const {
        switch (_intersectMethod) {
        case GPIntersectMethod::Mean:
            return intersectMean(sampler, ray, state, t);
        case GPIntersectMethod::MeanRaymarch:
            return intersectMeanRaymarch(sampler, ray, state, t);
        case GPIntersectMethod::GPDiscrete:
            return intersectGP(sampler, ray, state, t);
        default:
            return false;
        }
    }

    bool GaussianProcessMedium::intersectMean(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, float& t) const {
        t = ray.nearT() + 0.001f;
        for(int i = 0; i < 2048; i++) {
            auto p = ray.pos() + t * ray.dir();
            float m = (*_gp->_mean)(Derivative::None, p, Vec3f(0.f));

            if(m < 0.0001f) {
                auto ctxt = std::make_shared<GPContextFunctionSpace>();
                ctxt->derivs = { Derivative::None };
                ctxt->points = { p };
                ctxt->values = Eigen::MatrixXd(1, 1);
                ctxt->values(0, 0) = 0;
                state.gpContext = ctxt;
                return true;
            }

            t += m;

            if(t >= ray.farT()) {
                return false;
            }
        }
        return false;
    }

    bool GaussianProcessMedium::intersectMeanRaymarch(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, float& t) const {
        std::vector<Vec3f> points(_samplePoints);
        std::vector<Derivative> derivs(_samplePoints);
        std::vector<float> ts(_samplePoints);
        std::vector<float> vs(_samplePoints);
        float tOffset = sampler.next1D();
        for (int i = 0; i < _samplePoints; i++) {
            float t = lerp(ray.nearT(), min(100.f, ray.farT()), clamp((i - tOffset) / _samplePoints, 0.f, 1.f));
            ts[i] = t;
            points[i] = ray.pos() + t * ray.dir();
            vs[i] =  (*_gp->_mean)(Derivative::None, points[i], Vec3f(0.f));
        }

        float prevV = vs[0];
        float prevT = ts[0];
        for (int p = 1; p < _samplePoints; p++) {
            float currV = vs[p];
            float currT = ts[p];
            if (currV < 0) {
                float offsetT = prevV / (prevV - currV);
                t = lerp(prevT, currT, offsetT);

                auto ctxt = std::make_shared<GPContextFunctionSpace>();
                ctxt->derivs = { Derivative::None };
                ctxt->points = { ray.pos() + t * ray.dir() };
                ctxt->values = Eigen::MatrixXd(1, 1);
                ctxt->values(0, 0) = 0;
                state.gpContext = ctxt;
                return true;
            }
            prevV = currV;
            prevT = currT;
        }

        return false;
    }

    bool GaussianProcessMedium::intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, float& t) const {
        std::vector<Vec3f> points(_samplePoints);
        std::vector<Derivative> derivs(_samplePoints);
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
                cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
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

                points.resize(p + 1);
                derivs.resize(p + 1);

                points[p] = ray.pos() + t * ray.dir();
                gpSamples(p, 0) = 0;

                auto ctxt = std::make_shared<GPContextFunctionSpace>();
                if(_ctxt == GPCorrelationContext::Dori) {
                    ctxt->derivs = { Derivative::None };
                    ctxt->points = { ray.pos() + t * ray.dir() };
                    ctxt->values = Eigen::MatrixXd(1, 1);
                    ctxt->values(0, 0) = 0;
                } else {
                    ctxt->derivs = std::move(derivs);
                    ctxt->points = std::move(points);
                    ctxt->values = std::move(gpSamples);
                }
                state.gpContext = ctxt;
                return true;
            }
            prevV = currV;
            prevT = currT;
        }

        return false;
    }


    bool GaussianProcessMedium::sampleDistance(PathSampleGenerator & sampler, const Ray & ray,
        MediumState & state, MediumSample & sample) const
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
        }
        else {
            float t = maxT;

            if (intersect(sampler, ray, state, t)) {
                Vec3f ip = ray.pos() + ray.dir() * t;

                Vec3f grad;
                if (!sampleGradient(sampler, ray, ip, state, grad)) {
                    return false;
                }

                if (grad.dot(ray.dir()) > 0) {
                    //std::cout << "Gradient wrong direction: " << grad.dot(ray.dir()) << "\n";
                    return false;
                    //grad *= -1;
                }

                sample.aniso = grad;
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

                sample.exited = false;
            }
            else {
                sample.exited = true;
            }

            sample.t = min(t, maxT);
            sample.continuedT = t;
            sample.weight = sigmaS(ray.pos() + sample.t * ray.dir()) / sigmaT(ray.pos() + sample.t * ray.dir());
            sample.continuedWeight = sigmaS(ray.pos() + sample.continuedT * ray.dir()) / sigmaT(ray.pos() + sample.continuedT * ray.dir());
            sample.pdf = 1;

            state.lastAniso = sample.aniso;
            state.advance();
        }
        sample.p = ray.pos() + sample.t * ray.dir();
        sample.phase = _phaseFunction.get();

        return true;
    }

    Vec3f GaussianProcessMedium::transmittance(PathSampleGenerator & sampler, const Ray & ray, bool startOnSurface,
        bool endOnSurface, MediumSample * sample) const
    {
        if (ray.farT() == Ray::infinity())
            return Vec3f(0.0f);

        switch (_intersectMethod) {
            case GPIntersectMethod::GPDiscrete:
            {
                std::vector<Vec3f> points(_samplePoints);
                std::vector<Derivative> derivs(_samplePoints);

                for (int i = 0; i < points.size(); i++) {
                    float t = lerp(ray.nearT(), ray.farT(), (i + sampler.next1D()) / _samplePoints);
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
            case GPIntersectMethod::Mean:
            {
                MediumState state;
                state.firstScatter = startOnSurface;
                float t;
                return intersectMean(sampler, ray, state, t) ? Vec3f(0.f) : Vec3f(1.f);
            }
        }
    }

    float GaussianProcessMedium::pdf(PathSampleGenerator&/*sampler*/, const Ray & ray, bool startOnSurface, bool endOnSurface) const
    {
        return 1.0f;
    }
}
