#include "FunctionSpaceGaussianProcessMedium.hpp"

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

    FunctionSpaceGaussianProcessMedium::FunctionSpaceGaussianProcessMedium()
        : GaussianProcessMedium(
            std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(), std::make_shared<SquaredExponentialCovariance>()), 
            0.f, 0.f, 1.f, GPCorrelationContext::Goldfish, GPIntersectMethod::GPDiscrete, GPNormalSamplingMethod::ConditionedGaussian),
            _samplePoints(32),
            _stepSizeCov(0.)
    {
    }

    void FunctionSpaceGaussianProcessMedium::fromJson(JsonPtr value, const Scene& scene)
    {
        GaussianProcessMedium::fromJson(value, scene);
        value.getField("sample_points", _samplePoints);
        value.getField("step_size_cov", _stepSizeCov);
    }

    rapidjson::Value FunctionSpaceGaussianProcessMedium::toJson(Allocator& allocator) const
    {
        return JsonObject{ GaussianProcessMedium::toJson(allocator), allocator,
            "type", "function_space_gaussian_process",
            "sample_points", _samplePoints,
            "step_size_cov", _stepSizeCov,
        };
    }

    bool FunctionSpaceGaussianProcessMedium::intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const {
        std::vector<Vec3d> points(_samplePoints);
        std::vector<Derivative> derivs(_samplePoints);
        std::vector<double> ts(_samplePoints);
        double tOffset = sampler.next1D();

        auto ro = vec_conv<Vec3d>(ray.pos());
        auto rd = vec_conv<Vec3d>(ray.dir()).normalized();

        double maxRayDist = ray.farT() - ray.nearT();

        double determinedStepSize = maxRayDist / _samplePoints;

        if (_stepSizeCov > 0) {
            double goodStepSize = _gp->goodStepsize(ro, _stepSizeCov, rd);

            if (goodStepSize < determinedStepSize) {
                determinedStepSize = goodStepSize;
            }
        }

        maxRayDist = determinedStepSize * _samplePoints;


        double maxT = ray.nearT() + maxRayDist;

        for (int i = 0; i < _samplePoints; i++) {
            double rt = lerp((double)ray.nearT(), ray.nearT() + maxRayDist, clamp((i - tOffset) / (_samplePoints-1), 0., 1.));
            if (i == 0)
                rt = ray.nearT();
            else if (i == _samplePoints - 1)
                rt = ray.nearT() + maxRayDist;

            ts[i] =  rt;
            points[i] = ro + rt * rd;
            derivs[i] = Derivative::None;
        }

        Eigen::MatrixXd gpSamples;

        int startSign = 1;
        if (state.firstScatter) {
            std::array<Vec3d, 1> cond_pts = { points[0] };
            std::array<Derivative, 1> cond_deriv = { Derivative::None };
            std::array<double, 1> cond_vs = { _gp->sample_start_value(points[0], sampler) };
            std::array<Constraint, 1> constraints = { {0, 0, 0, FLT_MAX } };
            gpSamples = _gp->sample_cond(
                points.data(), derivs.data(), _samplePoints, nullptr,
                cond_pts.data(), cond_vs.data(), cond_deriv.data(), cond_pts.size(), nullptr,
                constraints.data(), constraints.size(),
                rd, 1, sampler);
        }
        else {
            auto ctxt = std::static_pointer_cast<GPContextFunctionSpace>(state.gpContext);

            if (ctxt->points.size() == 0) {
                std::cerr << "Empty context!\n";
            }

            assert(ctxt->points.size() > 0);

            Vec3d lastIntersectPt;
            double lastIntersectVal;

            if (ctxt->derivs[ctxt->points.size() - 1] == Derivative::None) {
                lastIntersectPt = ctxt->points[ctxt->points.size() - 1];
                lastIntersectVal = ctxt->values[ctxt->points.size() - 1];
            }
            else {
                lastIntersectPt = ctxt->points[ctxt->points.size() - 2];
                lastIntersectVal = ctxt->values[ctxt->points.size() - 2];
            }

            if (lastIntersectVal < 0) {
                std::cerr << "Conditioning on a value being less than zero: " << lastIntersectVal << "\n";
            }

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

                startSign = 1;
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

                double rayDeriv = state.lastAniso.dot(rd);

                std::array<double, 2> cond_vs = { lastIntersectVal, rayDeriv };

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

        if (prevV < -0.1) {
            std::cerr << "First sample along ray was way less than 0: " << prevV << "\n";
            return false;
        }

        prevV = max(prevV, 0.);

        double prevT = ts[0];
        for (int p = 1; p < _samplePoints; p++) {
            double currV = gpSamples(p, 0);
            double currT = ts[p];
            if (currV < 0) {
                double offsetT = prevV / (prevV - currV);
                t = lerp(prevT, currT, offsetT);

                if (t >= maxT) {
                    std::cerr << "Somehow got a distance that's greater than the max distance.\n";
                }

                derivs.resize(p + 2);
                points.resize(p + 2);
                gpSamples.conservativeResize(p + 2, Eigen::NoChange);

                points[p] = ro + t * rd;
                gpSamples(p, 0) = 0;
                derivs[p] = Derivative::None;

                points[p+1] = ro + t * rd;
                gpSamples(p+1, 0) = (prevV - currV) / (prevT - currT);
                derivs[p+1] = Derivative::First;

                auto ctxt = std::make_shared<GPContextFunctionSpace>();
                ctxt->derivs = std::move(derivs);
                ctxt->points = std::move(points);
                ctxt->values = std::vector<double>(gpSamples.data(), gpSamples.data() + ctxt->points.size());
                state.gpContext = ctxt;

                return true;
            }
            prevV = currV;
            prevT = currT;
        }

        t = maxT;
        auto ctxt = std::make_shared<GPContextFunctionSpace>();
        ctxt->derivs = std::move(derivs);
        ctxt->points = std::move(points);
        ctxt->values = std::vector<double>(gpSamples.data(), gpSamples.data() + ctxt->points.size());
        state.gpContext = ctxt;
        return false;
    }

    bool FunctionSpaceGaussianProcessMedium::sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state, Vec3d& grad) const {

        GPContextFunctionSpace& ctxt = *(GPContextFunctionSpace*)state.gpContext.get();

        auto rd = vec_conv<Vec3d>(ray.dir());

        switch (_normalSamplingMethod) {
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

            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(rd));

            if (ctxt.derivs[ctxt.points.size() - 1] == Derivative::None) {
                std::array<Vec3d, 3> gradDirs{
                    vec_conv<Vec3d>(frame.tangent),
                    vec_conv<Vec3d>(frame.bitangent),
                    vec_conv<Vec3d>(frame.normal)
                };

                auto gradSamples = _gp->sample_cond(
                    gradPs.data(), gradDerivs.data(), gradDirs.size(), gradDirs.data(),
                    ctxt.points.data(), ctxt.values.data(), ctxt.derivs.data(), ctxt.points.size(), nullptr,
                    nullptr, 0,
                    rd, 1, sampler);

                grad = vec_conv<Vec3d>(frame.toGlobal({
                    gradSamples(0,0), gradSamples(1,0), gradSamples(2,0)
                }));
            }
            else {
                std::array<Vec3d, 2> gradDirs{
                    vec_conv<Vec3d>(frame.tangent),
                    vec_conv<Vec3d>(frame.bitangent)
                };

                auto gradSamples = _gp->sample_cond(
                    gradPs.data(), gradDerivs.data(), gradDirs.size(), gradDirs.data(),
                    ctxt.points.data(), ctxt.values.data(), ctxt.derivs.data(), ctxt.points.size(), nullptr,
                    nullptr, 0,
                    rd, 1, sampler);

                grad = vec_conv<Vec3d>(frame.toGlobal({
                    gradSamples(0,0), gradSamples(1,0), ctxt.values[ctxt.points.size()-1]
                }));
            }

            if (!std::isfinite(grad.avg())) {
                std::cout << "Sampled gradient invalid.\n";
                return false;
            }

            break;
        }
        case GPNormalSamplingMethod::Beckmann:
        {
            Vec3d normal = _gp->_mean->dmean_da(ip).normalized();

            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

            Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
            float alpha = _gp->_cov->compute_beckmann_roughness();
            BeckmannNDF ndf(0, alpha, alpha);

            grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));
            break;
        }
        case GPNormalSamplingMethod::GGX:
        {
            Vec3d normal = _gp->_mean->dmean_da(ip).normalized();

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
    

    /*Vec3f FunctionSpaceGaussianProcessMedium::transmittance(PathSampleGenerator& sampler, const Ray& ray, bool startOnSurface,
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
                    std::array<Constraint, 1> constraints = { {0, 0, 0, FLT_MAX } };
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
    }*/
}
