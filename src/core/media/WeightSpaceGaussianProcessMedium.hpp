#ifndef WEIGHTSPACEGAUSSIANPROCESSMEDIUM_HPP_
#define WEIGHTSPACEGAUSSIANPROCESSMEDIUM_HPP_

#include "GaussianProcessMedium.hpp"
#include "math/WeightSpaceGaussianProcess.hpp"

namespace Tungsten {

class GaussianProcess;

struct GPContextWeightSpace : public GPContext {
    WeightSpaceRealization real;
};


class WeightSpaceGaussianProcessMedium : public GaussianProcessMedium
{
    int _numBasisFunctions;

public:

    WeightSpaceGaussianProcessMedium();
    WeightSpaceGaussianProcessMedium(std::shared_ptr<GaussianProcess> gp, 
        float materialSigmaA, float materialSigmaS, float density, int numBasisFunctions) :
        GaussianProcessMedium(gp, materialSigmaA, materialSigmaS, density), _numBasisFunctions(numBasisFunctions)
    {}

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual bool sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state,
        Vec3d& grad) const override;

    virtual bool intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const override;
};

}

#endif /* WEIGHTSPACEGAUSSIANPROCESSMEDIUM_HPP_ */
