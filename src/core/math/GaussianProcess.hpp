#ifndef GAUSSIANPROCESS_HPP_
#define GAUSSIANPROCESS_HPP_
#include "sampling/PathSampleGenerator.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "BitManip.hpp"
#include "math/Vec.hpp"
#include "math/MathUtil.hpp"
#include "math/Angle.hpp"
#include <functional>
#include <vector>

#include "io/JsonSerializable.hpp"
#include "io/JsonObject.hpp"
#include "primitives/Triangle.hpp"
#include "primitives/Vertex.hpp"

#include <autodiff/forward/real/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <igl/signed_distance.h>

using FloatD = autodiff::real2nd;

namespace fcpw {
    template<size_t DIM> class Scene;
}

namespace Tungsten {
    using Vec3Diff = autodiff::Vector3real2nd;
    using Vec4Diff = autodiff::Vector4real2nd;

    inline Vec3f from_diff(const Vec3Diff& vd) {
        return Vec3f{ (float)vd.x().val(), (float)vd.y().val(), (float)vd.z().val() };
    }

    inline Vec3Diff to_diff(const Vec3f& vd) {
        return Vec3Diff{ vd.x(), vd.y(), vd.z() };
    }

    template<typename Vec>
    inline auto dist2(Vec a, Vec b, Vec3f aniso) {
        auto d = b - a;
        return d.dot(Vec{ aniso.x(), aniso.y(), aniso.z() }.cwiseProduct(d));
    }

    class Grid;

    enum class Derivative : uint8_t {
        None = 0,
        First = 1
    };

    class CovarianceFunction : public JsonSerializable {

    public:
        float operator()(Derivative a, Derivative b, Vec3f pa, Vec3f pb, Vec3f gradDirA, Vec3f gradDirB) const {
            Vec3Diff pad = to_diff(pa);
            Vec3Diff pbd = to_diff(pb);
            Eigen::Array3d gradDirAD = Eigen::Array3d{ gradDirA.x(), gradDirA.y(), gradDirA.z() };
            Eigen::Array3d gradDirBD = Eigen::Array3d{ gradDirB.x(), gradDirB.y(), gradDirB.z() };

            if (a == Derivative::None && b == Derivative::None) {
                return cov(pa, pb);
            }
            else if (a == Derivative::First && b == Derivative::None) {
                return (float)dcov_da(pad, pbd, gradDirAD);
            }
            else if (a == Derivative::None && b == Derivative::First) {
                return (float)dcov_db(pad, pbd, gradDirBD);
            }
            else {
                return (float)dcov2_dadb(pad, pbd, gradDirAD, gradDirBD);
            }
        }

        virtual bool isMonotonic() const {
            return true;
        }

        virtual std::string id() const = 0;

        Vec3f _aniso;

    private:
        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const = 0;
        virtual float cov(Vec3f a, Vec3f b) const = 0;

        virtual FloatD dcov_da(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA) const;
        virtual FloatD dcov_db(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirB) const;
        virtual FloatD dcov2_dadb(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA, Eigen::Array3d dirB) const;

        friend class NonstationaryCovariance;
    };

    class NonstationaryCovariance : public CovarianceFunction {
    public:

        NonstationaryCovariance(
            std::shared_ptr<CovarianceFunction> stationaryCov = nullptr,
            std::shared_ptr<Grid> grid = nullptr,
            float offset = 0, float scale = 1) : _stationaryCov(stationaryCov), _grid(grid), _offset(offset), _scale(scale)
        {
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

        virtual std::string id() const {
            return tinyformat::format("ns-%s", _stationaryCov->id());
        }

    private:
        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override;
        virtual float cov(Vec3f a, Vec3f b) const override;

        FloatD sampleGrid(Vec3Diff a) const;
        autodiff::Matrix4real2nd _invGridTransformD;
        std::shared_ptr<CovarianceFunction> _stationaryCov;
        std::shared_ptr<Grid> _grid;
        float _offset;
        float _scale;
    };

    class SquaredExponentialCovariance : public CovarianceFunction {
    public:

        SquaredExponentialCovariance(float sigma = 1.f, float l = 1., Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _l(l) {
            _aniso = aniso;
        }

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

        virtual std::string id() const {
            return tinyformat::format("se/aniso=[%.1f,%.1f,%.1f]-s=%.3f-l=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l);
        }

    private:
        float _sigma, _l;

        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override {
            FloatD absq = dist2(a, b, _aniso);
            return sqr(_sigma)* exp(-(absq / (2 * sqr(_l))));
        }

        virtual float cov(Vec3f a, Vec3f b) const override {
            float absq = dist2(a, b, _aniso);
            return sqr(_sigma) * exp(-(absq / (2 * sqr(_l))));
        }
    };

    class RationalQuadraticCovariance : public CovarianceFunction {
    public:

        RationalQuadraticCovariance(float sigma = 1.f, float l = 1., float a = 1.0f, Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _l(l), _a(a) {
            _aniso = aniso;
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            CovarianceFunction::fromJson(value, scene);
            value.getField("sigma", _sigma);
            value.getField("a", _a);
            value.getField("lengthScale", _l);
            value.getField("aniso", _aniso);
        }

        virtual std::string id() const {
            return tinyformat::format("rq/aniso=[%.1f,%.1f,%.1f]-s=%.3f-l=%.3f-a=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l, _a);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "rational_quadratic",
                "sigma", _sigma,
                "a", _a,
                "lengthScale", _l,
                "aniso", _aniso
            };
        }
    private:
        float _sigma, _a, _l;

        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override {
            auto absq = dist2(a, b, _aniso);
            return sqr(_sigma) * pow((1.0f + absq / (2 * _a * _l * _l)), -_a);
        }

        virtual float cov(Vec3f a, Vec3f b) const override {
            auto absq = dist2(a, b, _aniso);
            return sqr(_sigma) * pow((1.0f + absq / (2 * _a * _l * _l)), -_a);
        }
    };

    class PeriodicCovariance : public CovarianceFunction {
    public:

        PeriodicCovariance(float sigma = 1.f, float l = 1., float w = TWO_PI, Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _l(l), _w(w) {
            _aniso = aniso;
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            CovarianceFunction::fromJson(value, scene);
            value.getField("sigma", _sigma);
            value.getField("w", _w);
            value.getField("lengthScale", _l);
            value.getField("aniso", _aniso);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "periodic",
                "sigma", _sigma,
                "w", _w,
                "lengthScale", _l,
                "aniso", _aniso
            };
        }

        virtual std::string id() const {
            return tinyformat::format("per/aniso=[%.1f,%.1f,%.1f]-s=%.3f-l=%.3f-w=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l, _w);
        }

    private:
        float _sigma, _w, _l;

        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override {
            auto absq = dist2(a, b, _aniso);
            return sqr(_sigma) * exp(-2 * pow(sin(PI * sqrt(absq) * _w), 2.f)) / (_l * _l);
        }

        virtual float cov(Vec3f a, Vec3f b) const override {
            auto absq = dist2(a, b, _aniso);
            return sqr(_sigma) * exp(-2 * pow(sin(PI * sqrt(absq) * _w), 2.f)) / (_l * _l);
        }
    };


    class ThinPlateCovariance : public CovarianceFunction {
    public:

        ThinPlateCovariance(float sigma = 1.f, float R = 1., Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _R(R) {
            _aniso = aniso;
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            CovarianceFunction::fromJson(value, scene);
            value.getField("sigma", _sigma);
            value.getField("R", _R);
            value.getField("aniso", _aniso);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "thin_plate",
                "sigma", _sigma,
                "R", _R,
                "aniso", _aniso
            };
        }

        virtual std::string id() const {
            return tinyformat::format("tp/aniso=[%.1f,%.1f,%.1f]-s=%.3f-R=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _R);
        }

    private:
        float _sigma, _R;

        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override {
            auto absq = dist2(a, b, _aniso);
            auto ab = sqrt(absq);
            return sqr(_sigma) / 12 * (2 * pow(ab, 3) - 3 * _R * absq + _R * _R * _R);
        }

        virtual float cov(Vec3f a, Vec3f b) const override {
            auto absq = dist2(a, b, _aniso);
            auto ab = sqrt(absq);
            return sqr(_sigma) / 12 * (2 * pow(ab, 3) - 3 * _R * absq + _R * _R * _R);
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

        TabulatedMean(std::shared_ptr<Grid> grid = nullptr, float offset = 0, float scale = 1) : _grid(grid), _offset(offset), _scale(scale) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

    private:
        std::shared_ptr<Grid> _grid;
        float _offset;
        float _scale;

        virtual float mean(Vec3f a) const override;
        virtual Vec3f dmean_da(Vec3f a) const override;
    };

    class MeshSdfMean : public MeanFunction {
    public:

        MeshSdfMean(PathPtr path = nullptr, bool isSigned = false) : _path(path), _signed(isSigned) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

    private:
        PathPtr _path;
        bool _signed;

        Eigen::MatrixXd V;
        Eigen::MatrixXi T, F;
        Eigen::MatrixXd FN, VN, EN;
        Eigen::MatrixXi E;
        Eigen::VectorXi EMAP;

        igl::FastWindingNumberBVH fwn_bvh;
        igl::AABB<Eigen::MatrixXd, 3> tree;

        Mat4f _configTransform;
        Mat4f _invConfigTransform;

        virtual float mean(Vec3f a) const override;
        virtual Vec3f dmean_da(Vec3f a) const override;
    };

#define SPARSE_COV

#ifdef SPARSE_COV
    using CovMatrix = Eigen::SparseMatrix<double>;
#else
    using CovMatrix = Eigen::MatrixXd;
#endif

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

        std::tuple<Eigen::VectorXd, CovMatrix> mean_and_cov(
            const Vec3f* points, const Derivative* derivative_types, const Vec3f* ddirs,
            Vec3f deriv_dir, size_t numPts) const;
        Eigen::VectorXd mean(
            const Vec3f* points, const Derivative* derivative_types, const Vec3f* ddirs,
            Vec3f deriv_dir, size_t numPts) const;

        CovMatrix cov(
            const Vec3f* points_a, const Vec3f* points_b,
            const Derivative* dtypes_a, const Derivative* dtypes_b,
            const Vec3f* ddirs_a, const Vec3f* ddirs_b,
            Vec3f deriv_dir, size_t numPtsA, size_t numPtsB) const;

        float sample_start_value(Vec3f p, PathSampleGenerator& sampler) const;

        Eigen::MatrixXd sample(
            const Vec3f* points, const Derivative* derivative_types, size_t numPts,
            const Vec3f* ddirs,
            const Constraint* constraints, size_t numConstraints,
            Vec3f deriv_dir, int samples, PathSampleGenerator& sampler) const;

        Eigen::MatrixXd sample_cond(
            const Vec3f* points, const Derivative* derivative_types, size_t numPts,
            const Vec3f* ddirs,
            const Vec3f* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
            const Vec3f* cond_ddirs,
            const Constraint* constraints, size_t numConstraints,
            Vec3f deriv_dir, int samples, PathSampleGenerator& sampler) const;


        void setConditioning(std::vector<Vec3f> globalCondPs, 
            std::vector<Derivative> globalCondDerivs, 
            std::vector<Vec3f> globalCondDerivDirs,
            std::vector<double> globalCondValues) {
            _globalCondPs = globalCondPs;
            _globalCondDerivs = globalCondDerivs;
            _globalCondDerivDirs = globalCondDerivDirs;
            _globalCondValues = globalCondValues;
        }

    public:
        // Get me some bits
        uint64_t vec2uint(Vec2f v) const;

        // From numpy
        double random_standard_normal(PathSampleGenerator& sampler) const;

        // Box muller transform
        Vec2d rand_normal_2(PathSampleGenerator& sampler) const;
        float rand_truncated_normal(float mean, float sigma, float a, PathSampleGenerator& sampler) const;

        Eigen::MatrixXd sample_multivariate_normal(
            const Eigen::VectorXd& mean, const CovMatrix& cov,
            const Constraint* constraints, int numConstraints,
            int samples, PathSampleGenerator& sampler) const;

        std::vector<Vec3f> _globalCondPs;
        std::vector<Derivative> _globalCondDerivs;
        std::vector<Vec3f> _globalCondDerivDirs;
        std::vector<double> _globalCondValues;

        PathPtr _conditioningDataPath;

        std::shared_ptr<MeanFunction> _mean;
        std::shared_ptr<CovarianceFunction> _cov;
        size_t _maxEigenvaluesN = 64;
        float _covEps = 0.0001f;
    };
}

#endif /* GAUSSIANPROCESS_HPP_ */