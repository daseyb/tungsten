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
#include "primitives/Triangle.hpp"
#include "primitives/Vertex.hpp"

#include <autodiff/forward/real/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using FloatD = autodiff::real2nd;

namespace fcpw {
    template<size_t DIM> class Scene;
}

namespace Tungsten {
    
    using Vec3Diff = autodiff::Vector3real2nd;

    inline Vec3f from_diff(const Vec3Diff& vd) {
        return Vec3f{ (float)vd.x(), (float)vd.y(), (float)vd.z() };
    }

    inline Vec3Diff to_diff(const Vec3f& vd) {
        return Vec3Diff{ vd.x(), vd.y(), vd.z() };
    }

    class Grid;

    enum class Derivative : uint8_t {
        None = 0,
        First = 1
    };

    class CovarianceFunction : public JsonSerializable {

    public:
        float operator()(Derivative a, Derivative b, Vec3f pa, Vec3f pb, Vec3f gradDir) const {

            Vec3Diff pad = to_diff(pa);
            Vec3Diff pbd = to_diff(pb);
            Eigen::Array3d gradDirD = Eigen::Array3d{ gradDir.x(), gradDir.y(), gradDir.z() };

            if (a == Derivative::None && b == Derivative::None) {
                return (float)cov(pad, pbd);
            }
            else if (a == Derivative::First && b == Derivative::None) {
                return (float)dcov_da(pad, pbd, gradDirD);
            }
            else if (a == Derivative::None && b == Derivative::First) {
                return (float)dcov_db(pad, pbd, gradDirD);
            }
            else {
                return (float)dcov2_dadb(pad, pbd, gradDirD);
            }
        }

    private:
        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const = 0;
        virtual FloatD dcov_da(Vec3Diff a, Vec3Diff b, Eigen::Array3d dir) const;
        virtual FloatD dcov_db(Vec3Diff a, Vec3Diff b, Eigen::Array3d dir) const;
        virtual FloatD dcov2_dadb(Vec3Diff a, Vec3Diff b, Eigen::Array3d dir) const;
    };

    class SquaredExponentialCovariance : public CovarianceFunction {
    public:

        SquaredExponentialCovariance(float sigma = 1.f, float l = 1., Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _l(l), _aniso(aniso) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            CovarianceFunction::fromJson(value, scene);
            value.getField("sigma", _sigma);
            value.getField("lengthScale", _l);
            value.getField("aniso", _aniso);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "squared_exponential",
                "sigma", _sigma,
                "lengthScale", _l,
                "aniso", _aniso
            };
        }
    private:
        float _sigma, _l;
        Vec3f _aniso;

        FloatD dist2(Vec3Diff a, Vec3Diff b) const {
            Vec3Diff d = b - a;
            return d.dot(to_diff(_aniso).cwiseProduct(d));
        }

        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override {
            FloatD absq = dist2(a, b);
            return sqr(_sigma)* exp(-(absq / (2 * sqr(_l))));
        }

        /*virtual FloatD dcov_da(Vec3Diff a, Vec3Diff b, Vec3Diff dir) const override {
            FloatD absq = dist2(a, b);
            FloatD ab = sqrt(absq);
            return ((exp(-(absq / (2 * sqr(_l)))) * ab * sqr(_sigma)) / sqr(_l));
        }

        virtual FloatD dcov_db(Vec3Diff a, Vec3Diff b, Vec3Diff dir) const override {
            return dcov_da(b, a);
        }

        virtual FloatD dcov2_dadb(Vec3Diff a, Vec3Diff b, Vec3Diff dir) const override {
            FloatD absq = dist2(a, b);
            return ((exp(-(absq / (2 * sqr(_l)))) * sqr(_sigma)) / sqr(_l) - (exp(-(absq / (2 * sqr(_l)))) * absq * sqr(_sigma)) / powf(_l, 4));
        }*/
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
            return (a - _c).length() - _r;
        }

        virtual Vec3f dmean_da(Vec3f a) const override {
            return (a - _c).normalized();
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


    class TabulatedMean : public MeanFunction {
    public:

        TabulatedMean(std::shared_ptr<Grid> grid = nullptr) : _grid(grid) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

    private:
        std::shared_ptr<Grid> _grid;

        virtual float mean(Vec3f a) const override;
        virtual Vec3f dmean_da(Vec3f a) const override;
    };

    class MeshSdfMean : public MeanFunction {
    public:

        MeshSdfMean(PathPtr path = nullptr) : _path(path) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

    private:
        PathPtr _path;
        std::shared_ptr<fcpw::Scene<3>> _scene;

        Mat4f _configTransform;
        Mat4f _invConfigTransform;

        std::vector<Vertex> _verts;
        std::vector<TriangleI> _tris;

        virtual float mean(Vec3f a) const override;
        virtual Vec3f dmean_da(Vec3f a) const override;
    };

    class GaussianProcess : public JsonSerializable {
    public:

        struct Constraint {
            int startIdx, endIdx;
            float minV, maxV;
        };

        GaussianProcess() : _mean(std::make_shared<HomogeneousMean>()), _cov(std::make_shared<SquaredExponentialCovariance>()) { }
        GaussianProcess(std::shared_ptr<MeanFunction> mean, std::shared_ptr<CovarianceFunction> cov) : _mean(mean), _cov(cov) { }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

        std::tuple<Eigen::VectorXf, Eigen::MatrixXf> mean_and_cov(const Vec3f* points, const Derivative* derivative_types, Vec3f deriv_dir, int numPts) const;
        Eigen::VectorXf mean(const Vec3f* points, const Derivative* derivative_types, Vec3f deriv_dir, int numPts) const;
        Eigen::MatrixXf cov(const Vec3f* points_a, const Vec3f* points_b, const Derivative* dtypes_a, const Derivative* dtypes_b, Vec3f deriv_dir, int numPtsA, int numPtsB) const;

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