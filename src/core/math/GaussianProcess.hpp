#ifndef GAUSSIANPROCESS_HPP_
#define GAUSSIANPROCESS_HPP_
#include "sampling/PathSampleGenerator.hpp"
#include "Eigen/Dense"
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

        SquaredExponentialCovariance(float sigma = 1.f, float l = 1.) : _sigma(sigma), _l(l) {}

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
            return ((exp(-(absq / (2 * sqr(_l)))) * ab * sqr(_sigma)) / sqr(_l));
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
        HomogeneousMean(float offset = 0.f) : _offset(offset) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            MeanFunction::fromJson(value, scene);
            value.getField("offset", _offset);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "homogeneous",
                "offset", _offset
            };
        }

    private:
        float _offset;

        virtual float mean(Vec3f a) const override {
            return _offset;
        }

        virtual Vec3f dmean_da(Vec3f a) const override {
            return Vec3f(0.f);
        }
    };

    class SphericalMean : public MeanFunction {
    public:

        SphericalMean(Vec3f c = Vec3f(0.f), float r = 1.) : _c(c), _r(r) {}

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

    class LinearMean : public MeanFunction {
    public:

        LinearMean(Vec3f ref = Vec3f(0.f), Vec3f dir = Vec3f(1.f, 0.f, 0.f), float scale = 1.0f) : 
            _ref(ref), _dir(dir.normalized()), _scale(scale) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            MeanFunction::fromJson(value, scene);

            value.getField("reference_point", _ref);
            value.getField("direction", _dir);
            value.getField("scale", _scale);

            _dir.normalize();
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "linear",
                "reference_point", _ref,
                "direction", _dir,
                "scale", _scale
            };
        }

    private:
        Vec3f _ref;
        Vec3f _dir;
        float _scale;

        virtual float mean(Vec3f a) const override {
            return (a - _ref).dot(_dir) * _scale;
        }

        virtual Vec3f dmean_da(Vec3f a) const override {
            return _dir * _scale;
        }
    };

    class GaussianProcess : public JsonSerializable {
    public:

        struct Constraint {
            int startIdx, endIdx;
            float minV, maxV;
        };

        GaussianProcess() : _mean(std::make_shared<HomogeneousMean>()), _cov(std::make_shared<SquaredExponentialCovariance>()) { }
        GaussianProcess(std::shared_ptr<MeanFunction> mean, std::shared_ptr<CovarianceFunction> cov) : _mean(mean), _cov(cov){ }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;

        std::tuple<Eigen::VectorXf, Eigen::MatrixXf> mean_and_cov(const Vec3f* points, const Derivative* derivative_types, Vec3f deriv_dir, int numPts) const;
        Eigen::VectorXf mean(const Vec3f* points, const Derivative* derivative_types, Vec3f deriv_dir, int numPts) const;
        Eigen::MatrixXf cov(const Vec3f* points_a, const Vec3f* points_b, const Derivative* dtypes_a, const Derivative* dtypes_b, int numPtsA, int numPtsB) const;

        float sample_start_value(Vec3f p, PathSampleGenerator& sampler) const;

        Eigen::MatrixXf sample(
            const Vec3f* points, const Derivative* derivative_types, int numPts,
            const Constraint* constraints, int numConstraints, 
            Vec3f deriv_dir, int samples, PathSampleGenerator& sampler) const;

        Eigen::MatrixXf sample_cond(
            const Vec3f* points, const Derivative* derivative_types, int numPts,
            const Vec3f* cond_points, const float* cond_values, const Derivative* cond_derivative_types, int numCondPts,
            const Constraint* constraints, int numConstraints,
            Vec3f deriv_dir, int samples, PathSampleGenerator& sampler) const;

    public:
        // Get me some bits
        uint64_t vec2uint(Vec2f v) const;

        // From numpy
        double random_standard_normal(PathSampleGenerator& sampler) const;

        // Box muller transform
        Vec2f rand_normal_2(PathSampleGenerator& sampler) const;
        float rand_truncated_normal(float mean, float sigma, float a, PathSampleGenerator& sampler) const;

        Eigen::MatrixXf sample_multivariate_normal(
            const Eigen::VectorXf& mean, const Eigen::MatrixXf& cov,
            const Constraint* constraints, int numConstraints,
            int samples, PathSampleGenerator& sampler) const;

        std::shared_ptr<MeanFunction> _mean;
        std::shared_ptr<CovarianceFunction> _cov;
    };
}

#endif /* GAUSSIANPROCESS_HPP_ */
