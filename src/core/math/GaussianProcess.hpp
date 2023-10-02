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

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include <math/AffineArithmetic.hpp>

#include <igl/signed_distance.h>

namespace Tungsten {
    using FloatDD = autodiff::dual2nd;
    using Vec3DD = autodiff::Vector3dual2nd;
    using Mat3DD = autodiff::Matrix3dual2nd;

    using FloatD = autodiff::real2nd;
    using Vec3Diff = autodiff::Vector3real2nd;
    using Vec4Diff = autodiff::Vector4real2nd;
    using Mat3Diff = autodiff::Matrix3real2nd;

    inline Vec3d from_diff(const Vec3DD& vd) {
        return Vec3d{ vd.x().val.val, vd.y().val.val, vd.z().val.val };
    }

    inline Vec3d from_diff(const Vec3Diff& vd) {
        return Vec3d{ vd.x().val(), vd.y().val(), vd.z().val() };
    }

    inline Vec3Diff to_diff(const Vec3f& vd) {
        return Vec3Diff{ vd.x(), vd.y(), vd.z() };
    }

    inline Vec3Diff to_diff(const Vec3d& vd) {
        return Vec3Diff{ vd.x(), vd.y(), vd.z() };
    }

    template<typename Vec>
    inline auto dist2(Vec a, Vec b, Vec3f aniso) {
        auto d = b - a;
        return d.dot(Vec{ aniso.x(), aniso.y(), aniso.z() }.cwiseProduct(d));
    }

    // Box muller transform
    Vec2d rand_normal_2(PathSampleGenerator& sampler);
    double rand_truncated_normal(double mean, double sigma, double a, PathSampleGenerator& sampler);
    Eigen::VectorXd sample_standard_normal(int n, PathSampleGenerator& sampler);


    double rand_gamma(double shape, double mean, PathSampleGenerator& samples);

    class Grid;
    class MeanFunction;

    enum class Derivative : uint8_t {
        None = 0,
        First = 1
    };

    class CovarianceFunction : public JsonSerializable {

    public:
        double operator()(Derivative a, Derivative b, Vec3d pa, Vec3d pb, Vec3d gradDirA, Vec3d gradDirB) const {
            if (a == Derivative::None && b == Derivative::None) {
                return cov(pa, pb);
            }
            else if (a == Derivative::First && b == Derivative::None) {
                return (double)dcov_da(to_diff(pa), to_diff(pb), vec_conv<Eigen::Array3d>(gradDirA));
            }
            else if (a == Derivative::None && b == Derivative::First) {
                return (double)dcov_db(to_diff(pa), to_diff(pb), vec_conv<Eigen::Array3d>(gradDirB));
            }
            else {
                return (double)dcov2_dadb(vec_conv<Vec3DD>(pa), vec_conv<Vec3DD>(pb), vec_conv<Eigen::Array3d>(gradDirA), vec_conv<Eigen::Array3d>(gradDirB));
            }
        }

        double spectral_density(Derivative a, Derivative b, double s) {
            if (a == Derivative::None && b == Derivative::None) {
                return spectral_density(s);
            }
            return 0.;
        }

        virtual bool hasAnalyticSpectralDensity() const { return false; }
        virtual double spectral_density(double s) const;
        virtual double sample_spectral_density(PathSampleGenerator& sampler) const;
        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler) const;

        virtual void loadResources() override;

        virtual bool isMonotonic() const { return true; }

        double compute_beckmann_roughness(Vec3d p = Vec3d(0.)) {
            double L2 = (*this)(Derivative::First, Derivative::First, p, p, Vec3d(1., 0., 0.), Vec3d(1., 0., 0.));
            return sqrt(2 * L2);
        }

        double compute_rices_formula(Vec3d p = Vec3d(0.), double u = 0.) {
            float L0 = (*this)(Derivative::None, Derivative::None, p, p, Vec3d(1., 0., 0.), Vec3d(1., 0., 0.));
            float L2 = (*this)(Derivative::First, Derivative::First, p, p, Vec3d(1., 0., 0.), Vec3d(1., 0., 0.));
            return exp(-u * u / (2 * L0)) * sqrt(L2 / L0) / (2 * PI);
        }

        virtual std::string id() const = 0;

        Vec3f _aniso = Vec3f(1.f);

        std::vector<double> discreteSpectralDensity;

    private:
        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const = 0;
        virtual FloatDD cov(Vec3DD a, Vec3DD b) const = 0;
        virtual double cov(Vec3d a, Vec3d b) const = 0;

        virtual FloatD dcov_da(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA) const;
        virtual FloatD dcov_db(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirB) const;
        virtual FloatDD dcov2_dadb(Vec3DD a, Vec3DD b, Eigen::Array3d dirA, Eigen::Array3d dirB) const;
    };

    class StationaryCovariance : public CovarianceFunction {
    private:
        virtual FloatD cov(FloatD absq) const = 0;
        virtual FloatDD cov(FloatDD absq) const = 0;
        virtual double cov(double absq) const = 0;

        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override {
            FloatD absq = dist2(a, b, _aniso);
            return cov(absq);
        }

        virtual FloatDD cov(Vec3DD a, Vec3DD b) const override {
            FloatDD absq = dist2(a, b, _aniso);
            return cov(absq);
        }

        virtual double cov(Vec3d a, Vec3d b) const override {
            double absq = dist2(a, b, _aniso);
            return cov(absq);
        }
      
        friend class NonstationaryCovariance;
        friend class MeanGradNonstationaryCovariance;
    };

    class MeanGradNonstationaryCovariance : public CovarianceFunction {
    public:

        MeanGradNonstationaryCovariance(
            std::shared_ptr<StationaryCovariance> stationaryCov = nullptr,
            std::shared_ptr<MeanFunction> mean = nullptr) : _stationaryCov(stationaryCov), _mean(mean)
        {
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

        virtual std::string id() const {
            return tinyformat::format("mean-ns-%s", _stationaryCov->id());
        }

        Eigen::Matrix3d localAniso(Vec3d p) const;

    private:
        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override;
        virtual FloatDD cov(Vec3DD a, Vec3DD b) const override;
        virtual double cov(Vec3d a, Vec3d b) const override;

        std::shared_ptr<StationaryCovariance> _stationaryCov;
        std::shared_ptr<MeanFunction> _mean;
    };

    class NonstationaryCovariance : public CovarianceFunction {
    public:

        NonstationaryCovariance(
            std::shared_ptr<StationaryCovariance> stationaryCov = nullptr,
            std::shared_ptr<Grid> variance = nullptr,
            std::shared_ptr<Grid> aniso = nullptr,
            float offset = 0, float scale = 1) : _stationaryCov(stationaryCov), _variance(variance), _aniso(aniso), _offset(offset), _scale(scale)
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
        virtual FloatDD cov(Vec3DD a, Vec3DD b) const override;
        virtual double cov(Vec3d a, Vec3d b) const override;

        FloatD sampleGrid(Vec3Diff a) const;
        FloatDD sampleGrid(Vec3DD a) const;

        std::shared_ptr<StationaryCovariance> _stationaryCov;
        std::shared_ptr<Grid> _variance;
        std::shared_ptr<Grid> _aniso;
        float _offset;
        float _scale;
    };


    
    class SquaredExponentialCovariance : public StationaryCovariance {
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
            return tinyformat::format("se/aniso=[%.4f,%.4f,%.4f]-s=%.3f-l=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l);
        }

        virtual bool hasAnalyticSpectralDensity() const override { return true; }
        virtual double spectral_density(double s) const {
            double norm = 1.0 / (sqrt(PI / 2) * sqr(_sigma));
            return norm * (exp(-0.5 * _l * _l * s * s) * _sigma * _sigma) / sqrt(1. / (_l * _l));
        }

        virtual double sample_spectral_density(PathSampleGenerator& sampler) const override {
            return abs(rand_normal_2(sampler).x() / _l);
        }

        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler) const override {
            double rad = (sqrt(2) * sqrt(-log(sampler.next1D()))) / _l;
            double angle = sampler.next1D() * 2 * PI;
            return Vec2d(sin(angle), cos(angle)) * rad;
        }

    private:
        float _sigma, _l;

        virtual FloatD cov(FloatD absq) const override {
            return sqr(_sigma) * exp(-(absq / (2 * sqr(_l))));
        }

        virtual FloatDD cov(FloatDD absq) const override {
            return sqr(_sigma) * exp(-(absq / (2 * sqr(_l))));
        }

        virtual double cov(double absq) const override {
            return sqr(_sigma) * exp(-(absq / (2 * sqr(_l))));
        }
    };

    class RationalQuadraticCovariance : public StationaryCovariance {
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
            return tinyformat::format("rq/aniso=[%.4f,%.4f,%.4f]-s=%.3f-l=%.3f-a=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l, _a);
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

        virtual bool hasAnalyticSpectralDensity() const override { return true; }
        virtual double spectral_density(double s) const {
            double norm = 1.0 / (sqrt(PI / 2.) * sqr(_sigma));
            return norm * (pow(2., 5. / 4. - _a / 2) * pow(1 / (_a * _l * _l), -(1. / 4.) - _a /
                2) * _sigma * _sigma * pow(abs(s), -0.5 + _a) *
                std::cyl_bessel_k(0.5 - _a, (sqrt(2) * abs(s)) / sqrt(1. / (_a * _l * _l)))) / std::tgamma(_a);
        }

        virtual double sample_spectral_density(PathSampleGenerator& sampler) const override {
            double tau = rand_gamma(_a, 1 / (_l * _l), sampler);
            double l = 1 / sqrt(tau);
            return abs(rand_normal_2(sampler).x() / l);
        }

        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler) const override {
            double rad = 2 * sqrt(rand_gamma(_a, 0.5 / (_l * _l), sampler)) * sqrt(-log(sampler.next1D()));
            double angle = sampler.next1D() * 2 * PI;
            return Vec2d(sin(angle), cos(angle)) * rad;
        }
    private:
        float _sigma, _a, _l;

        virtual FloatD cov(FloatD absq) const override {
            return sqr(_sigma) * pow((1.0f + absq / (2 * _a * _l * _l)), -_a);
        }

        virtual FloatDD cov(FloatDD absq) const override {
            return sqr(_sigma) * pow((1.0f + absq / (2 * _a * _l * _l)), -_a);
        }

        virtual double cov(double absq) const override {
            return sqr(_sigma) * pow((1.0f + absq / (2 * _a * _l * _l)), -_a);
        }
    };

    class MaternCovariance : public StationaryCovariance {
    public:

        MaternCovariance(float sigma = 1.f, float l = 1., float v = 1.0f, Vec3f aniso = Vec3f(1.f)) : _sigma(sigma), _l(l), _v(v) {
            _aniso = aniso;
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            CovarianceFunction::fromJson(value, scene);
            value.getField("sigma", _sigma);
            value.getField("v", _v);
            value.getField("lengthScale", _l);
            value.getField("aniso", _aniso);
        }

        virtual std::string id() const {
            return tinyformat::format("mat/aniso=[%.4f,%.4f,%.4f]-s=%.3f-l=%.3f-v=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l, _v);
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "matern",
                "sigma", _sigma,
                "v", _v,
                "lengthScale", _l,
                "aniso", _aniso
            };
        }

        virtual bool hasAnalyticSpectralDensity() const override { return true; }
        virtual double spectral_density(double s) const {
            const int D = 1;
            return pow(2, D) * pow(PI, D / 2.) * std::lgamma(_v + D / 2.) * 
                pow(2 * _v, _v) / (std::lgamma(_v) * pow(_l, 2 * _v)) * pow(2 * _v / sqr(_l) + 4 * sqr(PI) * sqr(s), -(_v + D / 2.));
        }

        virtual double sample_spectral_density(PathSampleGenerator& sampler) const override {
            return sqrt(-_v + _v * pow(1 - sampler.next1D(), -1. / _v)) / (sqrt(2) * _l * PI) * sin(PI * sampler.next1D());
        }

        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler) const override {
            double r = sqrt(-_v + _v * pow(1 - sampler.next1D(), -1. / _v)) / (sqrt(2) * _l * PI);
            double angle = sampler.next1D() * 2 * PI;
            return Vec2d(sin(angle), cos(angle)) * r;
        }

    private:
        float _sigma, _v, _l;

        virtual FloatD cov(FloatD absq) const override {
            assert(false);
            //auto r_scl = sqrt(2 * _v * absq) / _l;
            //return pow(2, 1 - _v) / std::lgamma(_v) * pow(r_scl, _v) * std::cyl_bessel_k(_v, r_scl);
            return 0;
        }

        virtual FloatDD cov(FloatDD absq) const override {
            assert(false);
            //auto r_scl = sqrt(2 * _v * absq) / _l;
            //return pow(2, 1 - _v) / std::lgamma(_v) * pow(r_scl, _v) * std::cyl_bessel_k(_v, r_scl);
            return 0;
        }

        virtual double cov(double absq) const override {
            if (absq < 0.0000001) {
                return sqr(_sigma);
            }
            auto r_scl = sqrt(2. * _v * absq) / _l;
            return sqr(_sigma) * pow(2, 1. - _v) / std::lgamma(_v) * pow(r_scl, _v) * std::cyl_bessel_k(_v, r_scl);
        }
    };

    class PeriodicCovariance : public StationaryCovariance {
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
            return tinyformat::format("per/aniso=[%.4f,%.4f,%.4f]-s=%.3f-l=%.3f-w=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _l, _w);
        }

    private:
        float _sigma, _w, _l;

        virtual FloatD cov(FloatD absq) const override {
            return sqr(_sigma) * exp(-2 * pow(sin(PI * sqrt(absq) * _w), 2.f)) / (_l * _l);
        }

        virtual FloatDD cov(FloatDD absq) const override {
            return sqr(_sigma) * exp(-2 * pow(sin(PI * sqrt(absq) * _w), 2.f)) / (_l * _l);
        }

        virtual double cov(double absq) const override {
            return sqr(_sigma) * exp(-2 * pow(sin(PI * sqrt(absq) * _w), 2.f)) / (_l * _l);
        }
    };


    class ThinPlateCovariance : public StationaryCovariance {
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
            return tinyformat::format("tp/aniso=[%.4f,%.4f,%.4f]-s=%.3f-R=%.3f", _aniso.x(), _aniso.y(), _aniso.z(), _sigma, _R);
        }

    private:
        float _sigma, _R;

        virtual FloatD cov(FloatD absq) const override {
            auto ab = sqrt(absq);
            return sqr(_sigma) / 12 * (2 * pow(ab, 3) - 3 * _R * absq + _R * _R * _R);
        }

        virtual FloatDD cov(FloatDD absq) const override {
            auto ab = sqrt(absq);
            return sqr(_sigma) / 12 * (2 * pow(ab, 3) - 3 * _R * absq + _R * _R * _R);
        }

        virtual double cov(double absq) const override {
            auto ab = sqrt(absq);
            return sqr(_sigma) / 12 * (2 * pow(ab, 3) - 3 * _R * absq + _R * _R * _R);
        }
    };

    class MeanFunction : public JsonSerializable {
    public:
        double operator()(Derivative a, Vec3d p, Vec3d d) const {
            if (a == Derivative::None) {
                return mean(p);
            }
            else {
                return d.dot(dmean_da(p));
            }
        }

        virtual Vec3d dmean_da(Vec3d a) const = 0;

        virtual Affine<1> mean(const Affine<3>& a) const {
            assert(false && "Not implemented!");
            return Affine<1>(0.);
        }

        virtual double lipschitz() const {
            return 1.;
        }
    private:
        virtual double mean(Vec3d a) const = 0;
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

        virtual double lipschitz() const override {
            return 0.;
        }

        virtual Affine<1> mean(const Affine<3>& a) const override {
            return Affine<1>(_offset);
        }

    private:
        float _offset;

        virtual double mean(Vec3d a) const override {
            return _offset;
        }

        virtual Vec3d dmean_da(Vec3d a) const override {
            return Vec3d(0.);
        }
    };

    class SphericalMean : public MeanFunction {
    public:

        SphericalMean(Vec3d c = Vec3d(0.), float r = 1.) : _c(c), _r(r) {}

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

        virtual Affine<1> mean(const Affine<3>& a) const override {
            return (a - _c).length() - _r;
        }

    private:
        Vec3d _c;
        float _r;

        virtual double mean(Vec3d a) const override {
            return (a - _c).length() - _r;
        }

        virtual Vec3d dmean_da(Vec3d a) const override {
            return (a - _c).normalized();
        }
    };

    class LinearMean : public MeanFunction {
    public:

        LinearMean(Vec3d ref = Vec3d(0.), Vec3d dir = Vec3d(1., 0., 0.), float scale = 1.0f) :
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

        virtual Affine<1> mean(const Affine<3>& a) const override {
            return dot(_dir, a - _ref) * _scale;
        }

    private:
        Vec3d _ref;
        Vec3d _dir;
        float _scale;

        virtual double mean(Vec3d a) const override {
            return (a - _ref).dot(_dir) * _scale;
        }

        virtual Vec3d dmean_da(Vec3d a) const override {
            return _dir * _scale;
        }

        virtual double lipschitz() const {
            return _scale * _dir.length();
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

        virtual double mean(Vec3d a) const override;
        virtual Vec3d dmean_da(Vec3d a) const override;
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

        virtual double mean(Vec3d a) const override;
        virtual Vec3d dmean_da(Vec3d a) const override;
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
            const Vec3d* points, const Derivative* derivative_types, const Vec3d* ddirs,
            Vec3d deriv_dir, size_t numPts) const;
        Eigen::VectorXd mean(
            const Vec3d* points, const Derivative* derivative_types, const Vec3d* ddirs,
            Vec3d deriv_dir, size_t numPts) const;

        CovMatrix cov(
            const Vec3d* points_a, const Vec3d* points_b,
            const Derivative* dtypes_a, const Derivative* dtypes_b,
            const Vec3d* ddirs_a, const Vec3d* ddirs_b,
            Vec3d deriv_dir, size_t numPtsA, size_t numPtsB) const;

        double sample_start_value(Vec3d p, PathSampleGenerator& sampler) const;

        Eigen::MatrixXd sample(
            const Vec3d* points, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Constraint* constraints, size_t numConstraints,
            Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const;

        Eigen::MatrixXd sample_cond(
            const Vec3d* points, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Vec3d* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
            const Vec3d* cond_ddirs,
            const Constraint* constraints, size_t numConstraints,
            Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const;


        void setConditioning(std::vector<Vec3d> globalCondPs, 
            std::vector<Derivative> globalCondDerivs, 
            std::vector<Vec3d> globalCondDerivDirs,
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
        double noIntersectBound(Vec3d p = Vec3d(0.), double q = 0.9999) const;
        double goodStepsize(Vec3d p = Vec3d(0.), double targetCov = 0.95) const;

        std::vector<Vec3d> _globalCondPs;
        std::vector<Derivative> _globalCondDerivs;
        std::vector<Vec3d> _globalCondDerivDirs;
        std::vector<double> _globalCondValues;

        PathPtr _conditioningDataPath;

        std::shared_ptr<MeanFunction> _mean;
        std::shared_ptr<CovarianceFunction> _cov;
        size_t _maxEigenvaluesN = 64;
        float _covEps = 0.f;
    };

    Eigen::MatrixXd sample_multivariate_normal(
        const Eigen::VectorXd& mean, const CovMatrix& cov,
        const GaussianProcess::Constraint* constraints, int numConstraints,
        int samples, PathSampleGenerator& sampler);
}

#endif /* GAUSSIANPROCESS_HPP_ */