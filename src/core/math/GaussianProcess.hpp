#ifndef GAUSSIANPROCESS_HPP_
#define GAUSSIANPROCESS_HPP_
#include "sampling/PathSampleGenerator.hpp"
#include "Eigen/dense"
#include "BitManip.hpp"
#include "math/Vec.hpp"
#include "math/MathUtil.hpp"
#include <functional>
#include <vector>
#include "io/JsonSerializable.hpp"
#include "io/JsonObject.hpp"


namespace Tungsten {
    enum class Derivative : uint8_t {
        None = 0,
        First = 1
    };

    class CovarianceFunction : public JsonSerializable {

    public:
        float operator()(Derivative a, Derivative b, Vec3f pa, Vec3f pb) const {
            if (a == Derivative::None && b == Derivative::None) {
                return cov(pa, pb);
            } else if (a == Derivative::First && b == Derivative::None) {
                return dcov_da(pa,pb);
            } else if (a == Derivative::None && b == Derivative::First) {
                return dcov_db(pa, pb);
            } else {
                return dcov2_dadb(pa, pb);
            }
        }

    private:
        virtual float cov(Vec3f a, Vec3f b) const = 0;
        virtual float dcov_da(Vec3f a, Vec3f b) const = 0;
        virtual float dcov_db(Vec3f a, Vec3f b) const  = 0;
        virtual float dcov2_dadb(Vec3f a, Vec3f b) const = 0;

    };

    class SquaredExponentialCovariance : public CovarianceFunction {
    public:
        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            CovarianceFunction::fromJson(value, scene);
            value.getField("sigma", _sigma);
            value.getField("lengthScale", _l);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "squared_exponential",
                "sigma", _sigma,
                "lengthScale", _l,
            };
        }
    private:
        float _sigma, _l;
        virtual float cov(Vec3f a, Vec3f b) const override {
            return sqr(_sigma) * exp(-( (a-b).lengthSq() / (2 * sqr(_l))));
        }

        virtual float dcov_da(Vec3f a, Vec3f b) const override {
            float absq = (a - b).lengthSq();
            float ab = sqrtf(absq);
            return -((exp(-(absq / (2 * sqr(_l)))) * ab * sqr(_sigma)) / sqr(_l));
        }

        virtual float dcov_db(Vec3f a, Vec3f b) const override {
            return dcov_da(b, a);
        }

        virtual float dcov2_dadb(Vec3f a, Vec3f b) const override {
            float absq = (a - b).lengthSq();
            return (exp(-(absq / (2 * sqr(_l)))) * sqr(_sigma)) / sqr(_l) - (exp(-(absq / (2 * sqr(_l)))) * absq * sqr(_sigma)) / powf(_l, 4);
        }
    };

    class MeanFunction : public JsonSerializable {
    public:
        float operator()(Derivative a, Vec3f p, Vec3f d) const {
            if (a == Derivative::None) {
                return mean(p);
            }
            else {
                return d.dot(dmean_da(p));
            }
        }

    private:
        virtual float mean(Vec3f a) const = 0;
        virtual Vec3f dmean_da(Vec3f a) const = 0;
    };

    class HomogeneousMean : public MeanFunction {
    public:
        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            MeanFunction::fromJson(value, scene);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "homogeneous"
            };
        }

    private:
        virtual float mean(Vec3f a) const override {
            return 0;
        }

        virtual Vec3f dmean_da(Vec3f a) const override {
            return Vec3f(0.f);
        }
    };

    class SphericalMean : public MeanFunction {
    public:
        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            MeanFunction::fromJson(value, scene);
            value.getField("center", _c);
            value.getField("radius", _r);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "spherical",
                "center", _c,
                "radius", _r,
            };
        }

    private:
        Vec3f _c;

        float _r;
        virtual float mean(Vec3f a) const override {
            return (a-_c).length() - _r;
        }

        virtual Vec3f dmean_da(Vec3f a) const override {
            return (a-_c).normalized();
        }
    };

    class GaussianProcess : public JsonSerializable {

        // Get me some bits
        uint64_t vec2uint(Vec2f v);

        // From numpy
        double random_standard_normal(PathSampleGenerator& sampler);

        // Box muller transform
        Vec2f rand_normal_2(PathSampleGenerator& sampler);

        Eigen::MatrixXf sample_multivariate_normal(const Eigen::VectorXf& mean, const Eigen::MatrixXf& cov, int samples, PathSampleGenerator& sampler);

        std::shared_ptr<MeanFunction> _mean;
        std::shared_ptr<CovarianceFunction> _cov;

    public:
        GaussianProcess() : _mean(std::make_shared<HomogeneousMean>()), _cov(std::make_shared<SquaredExponentialCovariance>()) { }
        GaussianProcess(std::shared_ptr<MeanFunction> mean, std::shared_ptr<CovarianceFunction> cov) : _mean(mean), _cov(cov){ }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;

        Eigen::MatrixXf sample(const std::vector<Vec3f>& points, const std::vector<Derivative>& derivative_types,
            const std::vector<Vec3f>& cond_points, const std::vector<float>& cond_values, const std::vector<Derivative>& cond_derivative_types, Vec3f deriv_dir, int samples,
            PathSampleGenerator& sampler);

    };
}

#endif /* GAUSSIANPROCESS_HPP_ */
