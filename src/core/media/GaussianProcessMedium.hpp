#ifndef GAUSSIANPROCESSMEDIUM_HPP_
#define GAUSSIANPROCESSMEDIUM_HPP_

#include "Medium.hpp"


namespace Tungsten {

class GaussianProcess;

class GaussianProcessMedium : public Medium
{
    Vec3f _materialSigmaA, _materialSigmaS;
    float _density;

    Vec3f _sigmaA, _sigmaS;
    Vec3f _sigmaT;
    bool _absorptionOnly;

    int _samplePoints;

public:
    std::shared_ptr<GaussianProcess> _gp;
    GaussianProcessMedium();
    GaussianProcessMedium(std::shared_ptr<GaussianProcess> gp, 
        float materialSigmaA, float materialSigmaS, float density, int samplePoints) : 
        _gp(gp), _materialSigmaA(materialSigmaA), _materialSigmaS(materialSigmaS), _density(density), _samplePoints(samplePoints) {}

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;
    virtual void loadResources() override;

    virtual bool isHomogeneous() const override;

    virtual void prepareForRender() override;

    virtual Vec3f sigmaA(Vec3f p) const override;
    virtual Vec3f sigmaS(Vec3f p) const override;
    virtual Vec3f sigmaT(Vec3f p) const override;

    std::vector<MediumSample> sampleDistanceMult(PathSampleGenerator& sampler, const Ray& ray, Medium::MediumState& state) const;

    virtual bool sampleDistance(PathSampleGenerator &sampler, const Ray &ray,
            MediumState &state, MediumSample &sample) const override;
    virtual Vec3f transmittance(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface, bool endOnSurface, MediumSample* sample) const override;
    virtual float pdf(PathSampleGenerator &sampler, const Ray &ray, bool startOnSurface, bool endOnSurface) const override;

    Vec3f sigmaA() const { return _sigmaA; }
    Vec3f sigmaS() const { return _sigmaS; }
};

}

#endif /* GAUSSIANPROCESSMEDIUM_HPP_ */
