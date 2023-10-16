#include "WeightSpaceGaussianProcessMedium.hpp"

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


    WeightSpaceGaussianProcessMedium::WeightSpaceGaussianProcessMedium()
        : GaussianProcessMedium(
            std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(), std::make_shared<SquaredExponentialCovariance>()),
            0.0f, 0.0f, 1.f),
          _numBasisFunctions(300)
    {
    }

    void WeightSpaceGaussianProcessMedium::fromJson(JsonPtr value, const Scene& scene)
    {
        GaussianProcessMedium::fromJson(value, scene);
        value.getField("basis_functions", _numBasisFunctions);
    }

    rapidjson::Value WeightSpaceGaussianProcessMedium::toJson(Allocator& allocator) const
    {
        return JsonObject{ GaussianProcessMedium::toJson(allocator), allocator,
            "type", "weight_space_gaussian_process",
            "basis_functions", _numBasisFunctions,
        };
    }

    bool WeightSpaceGaussianProcessMedium::sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state, Vec3d& grad) const {

        GPContextWeightSpace& ctxt = *(GPContextWeightSpace*)state.gpContext.get();

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

                std::array<double, 6> gradVs;
                for(int i = 0; i < 6; i++) {
                    gradVs[i] = ctxt.real.evaluate(gradPs[i]);
                }

                grad = Vec3d{
                    gradVs[0] - gradVs[3],
                    gradVs[1] - gradVs[4],
                    gradVs[2] - gradVs[5],
                } / (2 * eps);

                break;
            }
            case GPNormalSamplingMethod::ConditionedGaussian:
            {
                grad = ctxt.real.evaluateGradient(ip);
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

        return true;
    }

    bool WeightSpaceGaussianProcessMedium::intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const {
        if(state.firstScatter) {
            WeightSpaceBasis basis = WeightSpaceBasis::sample(_gp->_cov, _numBasisFunctions, sampler);
            
            auto ctxt = std::make_shared<GPContextWeightSpace>();
            ctxt->real = WeightSpaceRealization::sample(std::make_shared<WeightSpaceBasis>(basis), _gp, sampler);
            state.gpContext = ctxt;
        }

        GPContextWeightSpace& ctxt = *(GPContextWeightSpace*)state.gpContext.get();
        const WeightSpaceRealization& real = ctxt.real;

        double farT = min(ray.farT(), 1000.f);

        const double sig_0 = (farT - ray.nearT()) * 0.1f;
        const double delta = 0.01;
        const double np = 1.5;
        const double nm = 0.5;

        t = 0;
        double sig = sig_0;

        auto rd = vec_conv<Vec3d>(ray.dir());

        auto p = vec_conv<Vec3d>(ray.pos()) + (t + ray.nearT()) * rd;
        double f0 = real.evaluate(p);

        int sign0 = f0 < 0 ? -1 : 1;

        for (int i = 0; i < 2048 * 4; i++) {
            auto p_c = p + (t + ray.nearT() + delta) * rd;
            double f_c = real.evaluate(p_c);
            int signc = f_c < 0 ? -1 : 1;

            if (signc != sign0) {
                t += ray.nearT();
                return true;
            }

            auto c = p + (t + ray.nearT() + sig * 0.5) * rd;
            auto v = sig * 0.5 * rd;

            double nsig;
            if (real.rangeBound(c, { v }) != RangeBound::Unknown) {
                nsig = sig;
                sig = sig * np;
            }
            else {
                nsig = 0;
                sig = sig * nm;
            }

            t += max(nsig, delta);

            if (t + ray.nearT() >= farT) {
                return false;
            }
        }

        std::cerr << "Ran out of iterations in mean intersect IA." << std::endl;
        return false;
    }
}
