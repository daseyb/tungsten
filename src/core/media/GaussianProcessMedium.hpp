#ifndef GAUSSIANPROCESSMEDIUM_HPP_
#define GAUSSIANPROCESSMEDIUM_HPP_

#include "Medium.hpp"
#include "math/GaussianProcess.hpp"

namespace Tungsten {

class GaussianProcess;

struct GPContext {

};

struct GPContextWeightSpace : public GPContext {

};

struct GPContextFunctionSpace : public GPContext {
    std::vector<Vec3f> points;
    Eigen::MatrixXd values;
    std::vector<Derivative> derivs;
};

enum class GPCorrelationContext {
    Elephant,
    Goldfish,
    Dori
};

enum class GPIntersectMethod {
    Mean,
    MeanRaymarch,
    GPDiscrete
};

enum class GPNormalSamplingMethod {
    FiniteDifferences,
    ConditionedGaussian,
    Beckmann,
    GGX
};

class GaussianProcessMedium : public Medium
{
    Vec3f _materialSigmaA, _materialSigmaS;
    float _density;

    Vec3f _sigmaA, _sigmaS;
    Vec3f _sigmaT;
    bool _absorptionOnly;

    int _samplePoints;

    GPCorrelationContext _ctxt = GPCorrelationContext::Goldfish;
    GPIntersectMethod _intersectMethod = GPIntersectMethod::GPDiscrete;
    GPNormalSamplingMethod _normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian;

    static GPCorrelationContext stringToCorrelationContext(const std::string& name);
    static std::string correlationContextToString(GPCorrelationContext ctxt);

    static GPIntersectMethod stringToIntersectMethod(const std::string& name);
    static std::string intersectMethodToString(GPIntersectMethod ctxt);

    static GPNormalSamplingMethod stringToNormalSamplingMethod(const std::string& name);
    static std::string normalSamplingMethodToString(GPNormalSamplingMethod ctxt);

public:

    std::shared_ptr<GaussianProcess> _gp;
    GaussianProcessMedium();
    GaussianProcessMedium(std::shared_ptr<GaussianProcess> gp, 
        float materialSigmaA, float materialSigmaS, float density, int samplePoints,
        GPCorrelationContext ctxt = GPCorrelationContext::Goldfish, GPIntersectMethod intersectMethod = GPIntersectMethod::GPDiscrete, GPNormalSamplingMethod normalSamplingMethod = GPNormalSamplingMethod::ConditionedGaussian) :
        _gp(gp), _materialSigmaA(materialSigmaA), _materialSigmaS(materialSigmaS), _density(density), _samplePoints(samplePoints),
        _ctxt(ctxt), _intersectMethod(intersectMethod), _normalSamplingMethod(normalSamplingMethod)
    {}

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;
    virtual void loadResources() override;

    virtual bool isHomogeneous() const override;

    virtual void prepareForRender() override;

    virtual Vec3f sigmaA(Vec3f p) const override;
    virtual Vec3f sigmaS(Vec3f p) const override;
    virtual Vec3f sigmaT(Vec3f p) const override;

    bool sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3f& ip,
        MediumState& state,
        Vec3f& grad) const;

    bool intersect(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, float& t) const;
    bool intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, float& t) const;
    bool intersectMean(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, float& t) const;
    bool intersectMeanRaymarch(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, float& t) const;

    virtual bool sampleDistance(PathSampleGenerator &sampler, const Ray &ray,
            MediumState &state, MediumSample &sample) const override;
    virtual Vec3f transmittance(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface, bool endOnSurface, MediumSample* sample) const override;
    virtual float pdf(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface, bool endOnSurface) const override;

    Vec3f sigmaA() const { return _sigmaA; }
    Vec3f sigmaS() const { return _sigmaS; }
};

}

#endif /* GAUSSIANPROCESSMEDIUM_HPP_ */
