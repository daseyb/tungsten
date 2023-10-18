#ifndef FUNCTIONSPACEGAUSSIANPROCESSMEDIUM_HPP_
#define FUNCTIONSPACEGAUSSIANPROCESSMEDIUM_HPP_

#include "GaussianProcessMedium.hpp"

namespace Tungsten {

class GaussianProcess;

class FunctionSpaceGaussianProcessMedium : public GaussianProcessMedium
{
    int _samplePoints;

public:
    FunctionSpaceGaussianProcessMedium();
    FunctionSpaceGaussianProcessMedium(std::shared_ptr<GaussianProcess> gp, 
        float materialSigmaA, float materialSigmaS, float density, int samplePoints,
        GPCorrelationContext ctxt = GPCorrelationContext::Goldfish, GPIntersectMethod intersectMethod = GPIntersectMethod::GPDiscrete, GPNormalSamplingMethod normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian) : GaussianProcessMedium(gp, materialSigmaA, materialSigmaS, density, ctxt, intersectMethod, normalSamplingMethod), _samplePoints(samplePoints)
    {}

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual bool sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state,
        Vec3d& grad) const override;

    virtual bool intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const override;

    virtual Vec3f transmittance(PathSampleGenerator & sampler, const Ray & ray, bool startOnSurface,
        bool endOnSurface, MediumSample * sample) const override;
};

}

#endif /* FUNCTIONSPACEGAUSSIANPROCESSMEDIUM_HPP_ */
