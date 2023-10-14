#include "GaussianProcessMedium.hpp"

#include <cfloat>

#include "sampling/PathSampleGenerator.hpp"

#include "math/GaussianProcess.hpp"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "bsdfs/Microfacet.hpp"
#include <bsdfs/NDFs/beckmann.h>
#include <bsdfs/NDFs/GGX.h>

namespace Tungsten {

    std::string GaussianProcessMedium::correlationContextToString(GPCorrelationContext ctxt)
    {
        switch (ctxt) {
        default:
        case GPCorrelationContext::Elephant:  return "elephant";
        case GPCorrelationContext::Goldfish:   return "goldfish";
        case GPCorrelationContext::Dori:   return "dori";
        case GPCorrelationContext::None:   return "none";
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
        else if (name == "none")
            return GPCorrelationContext::None;
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
        case GPNormalSamplingMethod::GGX:   return "ggx";
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
        else if (name == "ggx")
            return GPNormalSamplingMethod::GGX;
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

    bool GaussianProcessMedium::sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state, Vec3d& grad) const {

        GPContextFunctionSpace& ctxt = *(GPContextFunctionSpace*)state.gpContext.get();

        auto rd = vec_conv<Vec3d>(ray.dir());

        switch(_normalSamplingMethod) {
            case GPNormalSamplingMethod::FiniteDifferences:
            {
                float eps = 0.0001f;
                std::array<Vec3d, 6> gradPs{
                    ip + Vec3d(eps, 0.f, 0.f),
                    ip + Vec3d(0.f, eps, 0.f),
                    ip + Vec3d(0.f, 0.f, eps),
                    ip - Vec3d(eps, 0.f, 0.f),
                    ip - Vec3d(0.f, eps, 0.f),
                    ip - Vec3d(0.f, 0.f, eps),
                };

                std::array<Derivative, 6> gradDerivs{
                    Derivative::None, Derivative::None, Derivative::None,
                    Derivative::None, Derivative::None, Derivative::None
                };

                Eigen::MatrixXd gradSamples = _gp->sample_cond(
                    gradPs.data(), gradDerivs.data(), gradPs.size(), nullptr,
                    ctxt.points.data(), ctxt.values.data(), ctxt.derivs.data(), ctxt.points.size(), nullptr,
                    nullptr, 0,
                    rd, 1, sampler);

                grad = Vec3d{
                    gradSamples(0,0) - gradSamples(3,0),
                    gradSamples(1,0) - gradSamples(4,0),
                    gradSamples(2,0) - gradSamples(5,0),
                } / (2 * eps);

                break;
            }
            case GPNormalSamplingMethod::ConditionedGaussian:
            {
                std::array<Vec3d, 3> gradPs{ ip, ip, ip };
                std::array<Derivative, 3> gradDerivs{ Derivative::First, Derivative::First, Derivative::First };
                std::array<Vec3d, 3> gradDirs{
                    Vec3d(1.f, 0.f, 0.f),
                    Vec3d(0.f, 1.f, 0.f),
                    Vec3d(0.f, 0.f, 1.f),
                };

                auto gradSamples = _gp->sample_cond(
                    gradPs.data(), gradDerivs.data(), gradPs.size(), gradDirs.data(),
                    ctxt.points.data(), ctxt.values.data(), ctxt.derivs.data(), ctxt.points.size(), nullptr,
                    nullptr, 0,
                    rd, 1, sampler);

                grad = {
                    gradSamples(0,0), gradSamples(1,0), gradSamples(2,0)
                };
                break;
            }
            case GPNormalSamplingMethod::Beckmann:
            {
                Vec3d normal = Vec3d(
                    _gp->_mean->operator()(Derivative::First, ip, Vec3d(1.f, 0.f, 0.f)),
                    _gp->_mean->operator()(Derivative::First, ip, Vec3d(0.f, 1.f, 0.f)),
                    _gp->_mean->operator()(Derivative::First, ip, Vec3d(0.f, 0.f, 1.f)));

                TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

                Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
                float alpha = _gp->_cov->compute_beckmann_roughness();
                BeckmannNDF ndf(0, alpha, alpha);

                grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));


                break;
            }
            case GPNormalSamplingMethod::GGX:
            {
                Vec3d normal = Vec3d(
                    _gp->_mean->operator()(Derivative::First, ip, Vec3d(1.f, 0.f, 0.f)),
                    _gp->_mean->operator()(Derivative::First, ip, Vec3d(0.f, 1.f, 0.f)),
                    _gp->_mean->operator()(Derivative::First, ip, Vec3d(0.f, 0.f, 1.f)));

                TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

                Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
                float alpha = _gp->_cov->compute_beckmann_roughness();
                GGXNDF ndf(0, alpha, alpha);
                grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));

                break;
            }
        }

        //grad *= state.startSign;

        return true;
    }

    bool GaussianProcessMedium::intersect(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const {
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

    bool GaussianProcessMedium::intersectMean(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const {
        t = ray.nearT() + 0.0001f;
        for(int i = 0; i < 2048*4; i++) {
            auto p = vec_conv<Vec3d>(ray.pos()) + t * vec_conv<Vec3d>(ray.dir());
            float m = (*_gp->_mean)(Derivative::None, p, Vec3d(0.f));

            if(m < 0.00001f) {
                auto ctxt = std::make_shared<GPContextFunctionSpace>();
                ctxt->derivs = { Derivative::None };
                ctxt->points = { p };
                ctxt->values = { 0. };
                state.gpContext = ctxt;
                return true;
            }

            t += m;

            if(t >= ray.farT()) {
                return false;
            }
        }

        std::cerr << "Ran out of iterations in mean intersect sphere trace." << std::endl;
        return false;
    }

    bool GaussianProcessMedium::intersectMeanRaymarch(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const {
        std::vector<Vec3d> points(_samplePoints);
        std::vector<Derivative> derivs(_samplePoints);
        std::vector<double> ts(_samplePoints);
        Eigen::MatrixXd gpSamples(_samplePoints,1);
        float tOffset = sampler.next1D();
        for (int i = 0; i < _samplePoints; i++) {
            double t = lerp(ray.nearT(), ray.nearT() + min(1000.f, ray.farT() - ray.nearT()), clamp((i + tOffset) / _samplePoints, 0.f, 1.f));
            ts[i] = t;
            points[i] = vec_conv<Vec3d>(ray.pos()) + t * vec_conv<Vec3d>(ray.dir());
            gpSamples(i,0) = (*_gp->_mean)(Derivative::None, points[i], Vec3d(0.));
        }

        double prevV = gpSamples(0, 0);
        double prevT = ts[0];
        for (int p = 1; p < _samplePoints; p++) {
            double currV = gpSamples(p, 0);
            double currT = ts[p];
            if (currV < 0) {
                double offsetT = prevV / (prevV - currV);
                t = lerp(prevT, currT, offsetT);

                derivs.resize(p + 1);
                points.resize(p + 1);

                points[p] = vec_conv<Vec3d>(ray.pos()) + t * vec_conv<Vec3d>(ray.dir());
                gpSamples(p, 0) = (*_gp->_mean)(Derivative::None, points[p], Vec3d(0.f));
               
                auto ctxt = std::make_shared<GPContextFunctionSpace>();
                if (_ctxt == GPCorrelationContext::Dori) {
                    ctxt->derivs = { Derivative::None };
                    ctxt->points = { points[p] };
                    ctxt->values = { gpSamples(p, 0) };
                }
                else {
                    ctxt->derivs = std::move(derivs);
                    ctxt->points = std::move(points);
                    ctxt->values = std::vector<double>(gpSamples.data(), gpSamples.data() + ctxt->points.size());
                }

                state.gpContext = ctxt;
                return true;
            }
            prevV = currV;
            prevT = currT;
        }

        return false;
    }

    bool GaussianProcessMedium::intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const {
        std::vector<Vec3d> points(_samplePoints);
        std::vector<Derivative> derivs(_samplePoints);
        std::vector<double> ts(_samplePoints);
        float tOffset = sampler.next1D();

        auto ro = vec_conv<Vec3d>(ray.pos());
        auto rd = vec_conv<Vec3d>(ray.dir());


        for (int i = 0; i < _samplePoints; i++) {
            double t = lerp(ray.nearT(), ray.nearT() + min(1000.f, ray.farT() - ray.nearT()), clamp((i - tOffset) / (_samplePoints-1), 0.f, 1.f));
            if (i == 0)
                t = ray.nearT();
            else if (i == _samplePoints - 1)
                t = ray.nearT() + min(1000.f, ray.farT() - ray.nearT());

            ts[i] =  t;
            points[i] = ro + t * rd;
            derivs[i] = Derivative::None;
        }

        Eigen::MatrixXd gpSamples;


        int startSign = 1;
        if (state.firstScatter) {
            std::array<Vec3d, 1> cond_pts = { points[0] };
            std::array<Derivative, 1> cond_deriv = { Derivative::None };
            std::array<double, 1> cond_vs = { _gp->sample_start_value(points[0], sampler) };
            std::array<GaussianProcess::Constraint, 1> constraints = { {0, 0, 0, FLT_MAX } };
            gpSamples = _gp->sample_cond(
                points.data(), derivs.data(), _samplePoints, nullptr,
                cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
                constraints.data(), constraints.size(),
                rd, 1, sampler);
        }
        else {
            auto ctxt = std::static_pointer_cast<GPContextFunctionSpace>(state.gpContext);

            assert(ctxt->points.size() > 0);

            auto lastIntersectPt = ctxt->points[ctxt->points.size() - 1];
            auto lastIntersectVal = ctxt->values[ctxt->points.size() - 1];

            switch (_ctxt) {
            case GPCorrelationContext::None:
            {
                startSign = 1; // state.lastAniso.dot(vec_conv<Vec3d>(ray.dir().normalized())) < 0 ? -1 : 1;
                gpSamples = _gp->sample(
                    points.data(), derivs.data(), _samplePoints, nullptr,
                    nullptr, 0,
                    rd, 1, sampler) * startSign;
                break;
            }
            case GPCorrelationContext::Dori:
            {
                std::array<Vec3d, 1> cond_pts = { lastIntersectPt };
                std::array<Derivative, 1> cond_deriv = { Derivative::None };
                std::array<double, 1> cond_vs = { lastIntersectVal };

                startSign = 1; // state.lastAniso.dot(vec_conv<Vec3d>(ray.dir().normalized())) < 0 ? -1 : 1;
                gpSamples = _gp->sample_cond(
                    points.data(), derivs.data(), _samplePoints, nullptr,
                    cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
                    nullptr, 0,
                    rd, 1, sampler) * startSign;
                break;
            }
            case GPCorrelationContext::Goldfish:
            {
                std::array<Vec3d, 2> cond_pts = { lastIntersectPt, lastIntersectPt };
                std::array<Derivative, 2> cond_deriv = { Derivative::None, Derivative::First };
                std::array<double, 2> cond_vs = { lastIntersectVal, state.lastAniso.dot(vec_conv<Vec3d>(ray.dir().normalized())) };

                startSign = 1;
                gpSamples = _gp->sample_cond(
                    points.data(), derivs.data(), _samplePoints, nullptr,
                    cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
                    nullptr, 0,
                    rd, 1, sampler) * startSign;
                break;
            }
            case GPCorrelationContext::Elephant:
            {
                //ctxt->points.erase(ctxt->points.begin(), ctxt->points.end() - 3);
                //ctxt->derivs.erase(ctxt->derivs.begin(), ctxt->derivs.end() - 3);
                //ctxt->values.erase(ctxt->values.begin(), ctxt->values.end() - 3);
                //ctxt->points.push_back(lastIntersectPt);
                //ctxt->derivs.push_back(Derivative::First);
                //ctxt->values.push_back(state.lastAniso.dot(vec_conv<Vec3d>(ray.dir().normalized())));

                /*std::array<Vec3f, 4> cond_pts = {points[0], points[0], points[0], points[0]};
                std::array<Derivative, 4> cond_deriv = { Derivative::None, Derivative::First, Derivative::First, Derivative::First };
                std::array<double, 4> cond_vs = { 0, state.lastAniso.x(), state.lastAniso.y(), state.lastAniso.z() };
                std::array<Vec3f, 4> cond_dirs = { ray.dir().normalized(), 
                    Vec3f(1.f, 0.f, 0.f), 
                    Vec3f(0.f, 1.f, 0.f), 
                    Vec3f(0.f, 0.f, 1.f) };*/ 

                startSign = 1;
                gpSamples = _gp->sample_cond(
                    points.data(), derivs.data(), _samplePoints, nullptr,
                    ctxt->points.data(), ctxt->values.data(), ctxt->derivs.data(), ctxt->points.size(), nullptr,
                    nullptr, 0,
                    rd, 1, sampler) * startSign;
                break;
            }
            }
        }

        double prevV = gpSamples(0, 0);

        if (state.firstScatter && prevV < 0) {
            return false;
        }

        double prevT = ts[0];
        for (int p = 1; p < _samplePoints; p++) {
            double currV = gpSamples(p, 0);
            double currT = ts[p];
            if (currV < 0) {
                double offsetT = prevV / (prevV - currV);
                t = lerp(prevT, currT, offsetT);

                derivs.resize(p + 2);
                points.resize(p + 2);
                gpSamples.conservativeResize(p + 2, Eigen::NoChange);

                points[p] = vec_conv<Vec3d>(ray.pos()) + t * vec_conv<Vec3d>(ray.dir());
                gpSamples(p, 0) = 0;
                derivs[p] = Derivative::None;

                points[p + 1] = vec_conv<Vec3d>(ray.pos()) + t * vec_conv<Vec3d>(ray.dir());
                gpSamples(p + 1, 0) = (prevV - currV) / (prevT - currT);
                derivs[p + 1] = Derivative::First;

                auto ctxt = std::make_shared<GPContextFunctionSpace>();
                /*if (_ctxt == GPCorrelationContext::Dori) {
                    ctxt->derivs = { Derivative::None };
                    ctxt->points = { points[p] };
                    ctxt->values = { gpSamples(p, 0) };
                } else*/ {
                    ctxt->derivs = std::move(derivs);
                    ctxt->points = std::move(points);
                    ctxt->values = std::vector<double>(gpSamples.data(), gpSamples.data() + ctxt->points.size());
                }
                state.gpContext = ctxt;
                return true;
            }
            prevV = currV;
            prevT = currT;
        }

        auto ctxt = std::make_shared<GPContextFunctionSpace>();
        ctxt->derivs = std::move(derivs);
        ctxt->points = std::move(points);
        ctxt->values = std::vector<double>(gpSamples.data(), gpSamples.data() + ctxt->points.size());
        state.gpContext = ctxt;
        return false;
    }


    bool GaussianProcessMedium::sampleDistance(PathSampleGenerator & sampler, const Ray & ray,
        MediumState & state, MediumSample & sample) const
    {
        sample.emission = Vec3f(0.0f);
        float maxT = ray.farT();

        if (state.bounce >= _maxBounce) {
            sample.t = maxT;
            sample.weight = Vec3f(1.f);
            sample.pdf = 1.0f;
            sample.exited = true;
            sample.p = ray.pos() + sample.t * ray.dir();
            sample.phase = _phaseFunction.get();
            return true;
        }

        if (_absorptionOnly) {
            if (maxT == Ray::infinity())
                return false;
            sample.t = maxT;
            sample.weight = transmittance(sampler, ray, state.firstScatter, true, &sample);
            sample.pdf = 1.0f;
            sample.exited = true;
        }
        else {
            double t = maxT;

            sample.exited = !intersect(sampler, ray, state, t);
            {
                Vec3d ip = vec_conv<Vec3d>(ray.pos()) + vec_conv<Vec3d>(ray.dir()) * t;

                Vec3d grad;
                if (!sampleGradient(sampler, ray, vec_conv<Vec3d>(ip), state, grad)) {
                    return false;
                }

                //if (!sample.exited && grad.dot(vec_conv<Vec3d>(ray.dir())) > 0) {
                //    return false;
                //}

                sample.aniso = grad;
                if (!std::isfinite(sample.aniso.avg())) {
                    sample.aniso = Vec3d(1.f, 0.f, 0.f);
                    std::cout << "Gradient invalid.\n";
                    return false;
                }

                if (sample.aniso.lengthSq() < 0.0000001f) {
                    sample.aniso = Vec3d(1.f, 0.f, 0.f);
                    std::cout << "Gradient zero.\n";
                    return false;
                }
            }

            sample.t = min(float(t), maxT);
            sample.continuedT = float(t);
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

        auto rd = vec_conv<Vec3d>(ray.dir());

        switch (_intersectMethod) {
            case GPIntersectMethod::GPDiscrete:
            {
                std::vector<Vec3d> points(_samplePoints);
                std::vector<Derivative> derivs(_samplePoints);

                for (int i = 0; i < points.size(); i++) {
                    float t = lerp(ray.nearT(), ray.farT(), (i + sampler.next1D()) / _samplePoints);
                    points[i] = vec_conv<Vec3d>(ray.pos() + t * ray.dir());
                    derivs[i] = Derivative::None;
                }

                Eigen::MatrixXd gpSamples;
                int startSign = 1;

                if (startOnSurface) {
                    std::array<Vec3d, 1> cond_pts = { points[0] };
                    std::array<Derivative, 1> cond_deriv = { Derivative::None };
                    std::array<double, 1> cond_vs = { _gp->sample_start_value(points[0], sampler) };
                    std::array<GaussianProcess::Constraint, 1> constraints = { {0, 0, 0, FLT_MAX } };
                    gpSamples = _gp->sample_cond(
                        points.data(), derivs.data(), points.size(), nullptr,
                        cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
                        constraints.data(), constraints.size(),
                        rd, 10, sampler);
                }
                else {
                    if (!sample) {
                        std::cout << "what\n";
                        return Vec3f(0.f);
                    }

                    std::array<Vec3d, 2> cond_pts = { points[0], points[0] };
                    std::array<Derivative, 2> cond_deriv = { Derivative::None, Derivative::First };

                    double deriv = sample->aniso.dot(vec_conv<Vec3d>(ray.dir().normalized()));
                    startSign = deriv < 0 ? -1 : 1;
                    std::array<double, 2> cond_vs = { 0, deriv };

                    gpSamples = _gp->sample_cond(
                        points.data(), derivs.data(), points.size(), nullptr,
                        cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
                        nullptr, 0,
                        rd, 10, sampler) * startSign;
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
                double t;
                return intersectMean(sampler, ray, state, t) ? Vec3f(0.f) : Vec3f(1.f);
            }
        }
    }

    float GaussianProcessMedium::pdf(PathSampleGenerator&/*sampler*/, const Ray & ray, bool startOnSurface, bool endOnSurface) const
    {
        return 1.0f;
    }
}
