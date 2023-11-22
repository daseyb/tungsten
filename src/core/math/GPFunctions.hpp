#ifndef GPFUNCTIONS_HPP_
#define GPFUNCTIONS_HPP_

#include <math/Vec.hpp>
#include <math/Angle.hpp>
#include "io/JsonSerializable.hpp"
#include "io/JsonObject.hpp"
#include <math/AffineArithmetic.hpp>
#include <sampling/SampleWarp.hpp>
#include <sampling/Gaussian.hpp>

#include <math/SdfFunctions.hpp>
#include <math/GPNeuralNetwork.hpp>

#include <autodiff/forward/real/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include <igl/fast_winding_number.h>
#include <igl/AABB.h>


namespace Tungsten {
    using FloatDD = autodiff::dual2nd;
    using Vec3DD = autodiff::Vector3dual2nd;
    using Mat3DD = autodiff::Matrix3dual2nd;
    using VecXDD = autodiff::VectorXdual2nd;

    using FloatD = autodiff::real2nd;
    using Vec3Diff = autodiff::Vector3real2nd;
    using Vec4Diff = autodiff::Vector4real2nd;
    using Mat3Diff = autodiff::Matrix3real2nd;
    using VecXDiff = autodiff::VectorXreal2nd;

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

    template <typename Mat, typename Vec>
    static inline Mat compute_ansio(const Vec& grad, const Vec& aniso) {
        TangentFrameD<Mat, Vec> tf(grad);

        auto vmat = tf.toMatrix();
        Mat smat = Mat::Identity();
        smat.diagonal() = aniso;

        return vmat * smat * vmat.transpose();
    }

    template<typename Vec>
    static inline Vec mult(const Eigen::Matrix2d& a, const Vec& b)
    {
        return Vec(
            a(0, 0) * b.x() + a(0, 1) * b.y(),
            a(1, 0) * b.x() + a(1, 1) * b.y()
        );
    }

    template<typename Vec>
    static inline Vec mult(const Eigen::Matrix3d& a, const Vec& b)
    {
        return Vec(
            a(0, 0) * b.x() + a(0, 1) * b.y() + a(0, 2) * b.z(),
            a(1, 0) * b.x() + a(1, 1) * b.y() + a(1, 2) * b.z(),
            a(2, 0) * b.x() + a(2, 1) * b.y() + a(2, 2) * b.z()
        );
    }

    template<typename Vec>
    static inline Vec mult(const Mat4f& a, const Vec& b)
    {
        return Vec(
            a(0, 0) * b.x() + a(0, 1) * b.y() + a(0, 2) * b.z() + a(0, 3),
            a(1, 0) * b.x() + a(1, 1) * b.y() + a(1, 2) * b.z() + a(1, 3),
            a(2, 0) * b.x() + a(2, 1) * b.y() + a(2, 2) * b.z() + a(2, 3)
        );
    }

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
        virtual bool requireProjection() const { return false; }
        virtual double spectral_density(double s) const;
        virtual double sample_spectral_density(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const;
        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const;
        virtual Vec3d sample_spectral_density_3d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const;

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
        friend class ProceduralNonstationaryCovariance;
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


    template<typename ElemType>
    static ElemType trilinearInterpolation(const Vec3d& uv, ElemType(&data)[2][2][2]) {
        return lerp(
            lerp(
                lerp(data[0][0][0], data[1][0][0], uv.x()),
                lerp(data[0][1][0], data[1][1][0], uv.x()),
                uv.y()
            ),
            lerp(
                lerp(data[0][0][1], data[1][0][1], uv.x()),
                lerp(data[0][1][1], data[1][1][1], uv.x()),
                uv.y()
            ),
            uv.z()
        );
    };

    template<typename ElemType>
    static ElemType bilinearInterpolation(const Vec2d& uv, ElemType(&data)[2][2]) {
        return lerp(
            lerp(data[0][0], data[1][0], uv.x()),
            lerp(data[0][1], data[1][1], uv.x()),
            uv.y()
        );
    };

    template<typename ElemType>
    struct RegularGrid : JsonSerializable {

        RegularGrid(Box3d bounds = Box3d(), size_t res = 0, std::vector<ElemType> values = {}) : bounds(bounds), res(res), values(values) {}

        Box3d bounds;
        size_t res;
        std::vector<ElemType> values;

        PathPtr path;

        ElemType getValue(Vec3i coord) const {
            coord = clamp(coord, Vec3i(0), Vec3i(res - 1));
            return values[ (coord.x() * res + coord.y()) * res + coord.z()];
        };

        void getValues(const Vec3i& coord, ElemType(&data)[2][2][2]) const {
            data[0][0][0] = getValue(coord + Vec3i(0, 0, 0));
            data[1][0][0] = getValue(coord + Vec3i(1, 0, 0));
            data[0][1][0] = getValue(coord + Vec3i(0, 1, 0));
            data[1][1][0] = getValue(coord + Vec3i(1, 1, 0));

            data[0][0][1] = getValue(coord + Vec3i(0, 0, 1));
            data[1][0][1] = getValue(coord + Vec3i(1, 0, 1));
            data[0][1][1] = getValue(coord + Vec3i(0, 1, 1));
            data[1][1][1] = getValue(coord + Vec3i(1, 1, 1));
        };

        ElemType getValue(Vec3d p) const {
            Vec3d p_grid = (double(res) * (p - bounds.min()) / bounds.diagonal()) - 0.5;
            ElemType data[2][2][2];
            getValues(Vec3i((int)floor(p_grid.x()), (int)floor(p_grid.y()), (int)floor(p_grid.z())), data);
            return trilinearInterpolation(
                Vec3d(p_grid.x() - floor(p_grid.x()), p_grid.y() - floor(p_grid.y()), p_grid.z() - floor(p_grid.z())), 
                data);
        }

        std::vector<Vec3d> makePoints(bool centered = false) const {
            std::vector<Vec3d> points(res * res * res);
            int idx = 0;
            for (int i = 0; i < res; i++) {
                for (int j = 0; j < res; j++) {
                    for (int k = 0; k < res; k++) {
                        if (centered) {
                            points[idx] = lerp(bounds.min(), bounds.max(), (Vec3d((double)i + 0.5, (double)j + 0.5, (double)k + 0.5) / res));
                        }
                        else {
                            points[idx] = lerp(bounds.min(), bounds.max(), (Vec3d((double)i, (double)j, (double)k) / (res - 1)));
                        }
                        idx++;
                    }
                }
            }
            return points;
        }

        virtual void saveResources() override {
            if (path) {
                std::ofstream xfile(path->absolute().asString(), std::ios::out | std::ios::binary);
                xfile.write((char*)values.data(), sizeof(values[0]) * values.size());
                xfile.close();
            }
        }


        virtual void loadResources() override {
            if (path) {
                std::ifstream xfile(path->absolute().asString(), std::ios::in | std::ios::binary);
                ElemType value;
                while (xfile.read(reinterpret_cast<char*>(&value), sizeof(ElemType))) {
                    values.push_back(value);
                }
                xfile.close();
            }
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "regular_grid",
                "bounds_min", bounds.min(),
                "bounds_max", bounds.max(),
                "res", res,
                "path", *path
            };
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            JsonSerializable::fromJson(value, scene);
            value.getField("bounds_min", bounds.min());
            value.getField("bounds_max", bounds.max());
            value.getField("res", res);

            if (auto f = value["path"]) path = scene.fetchResource(f);
        }
    };

    template<typename Elem>
    void save_grid(RegularGrid<Elem>& grid, Path path) {

        DirectoryChange context(path.parent());

        grid.path = std::make_shared<Path>(path.stripParent().stripExtension() + "-values.bin");
        grid.saveResources();

        rapidjson::Document document;
        document.SetObject();
        *(static_cast<rapidjson::Value*>(&document)) = grid.toJson(document.GetAllocator());
        FileUtils::writeJson(document, path.stripParent());
    }

    template<typename Elem>
    RegularGrid<Elem> load_grid(Path path) {
        JsonDocument document(path);

        DirectoryChange context(path.parent());

        Scene scene(path.parent(), nullptr);
        scene.setPath(path);

        RegularGrid<Elem> grid;
        grid.fromJson(document, scene);
        grid.loadResources();

        return grid;
    }

    class ProceduralScalar : public JsonSerializable {
    public:
        virtual double operator()(Vec3d p) const = 0;
    };

    class ConstantScalar : public ProceduralScalar {
        double _v;
    public:
        ConstantScalar(double v = 0) : _v(v) { }
        virtual double operator()(Vec3d p) const override {
            return _v;
        }
    };

    class RegularGridScalar : public ProceduralScalar {
        std::shared_ptr<RegularGrid<double>> _grid;
    public:
        RegularGridScalar(std::shared_ptr<RegularGrid<double>> grid) : _grid(grid) {}

        virtual double operator()(Vec3d p) const override {
            return _grid->getValue(p);
        }
    };



    class LinearRampScalar : public ProceduralScalar {
        Vec2d _minMax;
        Vec3d _dir;
        Vec2d _range;
    public:
        LinearRampScalar(Vec2d minMax, Vec3d dir, Vec2d range) : _minMax(minMax), _dir(dir), _range(range) { }

        virtual double operator()(Vec3d p) const override {
            double i = p.dot(_dir);
            i = clamp((i - _range.x()) / (_range.y() - _range.x()), 0., 1.);
            return lerp(_minMax.x(), _minMax.y(), i);
        }
    };

    class ProceduralVector : public JsonSerializable {
    public:
        virtual Vec3d operator()(Vec3d p) const = 0;
    };


    class ConstantVector : public ProceduralVector {
        Vec3d _v;
    public:
        ConstantVector(Vec3d v = Vec3d(0.)) : _v(v) { }
        virtual Vec3d operator()(Vec3d p) const override {
            return _v;
        }
    };

    class RegularGridVector : public ProceduralVector {
        std::shared_ptr<RegularGrid<Vec3d>> _grid;
    public:
        RegularGridVector(std::shared_ptr<RegularGrid<Vec3d>> grid) : _grid(grid) {}

        virtual Vec3d operator()(Vec3d p) const override {
            return _grid->getValue(p);
        }
    };

    class LinearRampVector : public ProceduralVector {
        Vec3d _min;
        Vec3d _max;
        Vec3d _dir;
        Vec2d _range;
    public:
        LinearRampVector(Vec3d min, Vec3d max, Vec3d dir, Vec2d range) : _min(min), _max(max), _dir(dir), _range(range) { }

        virtual Vec3d operator()(Vec3d p) const override {
            double i = p.dot(_dir);
            i = clamp((i - _range.x()) / (_range.y() - _range.x()), 0., 1.);
            return lerp(_min, _max, i);
        }
    };

    class ProceduralScalarCode : public ProceduralScalar {
        std::function<double(Vec3d)> _fn;
    public:
        ProceduralScalarCode(std::function<double(Vec3d)> fn = nullptr) : _fn(fn) {}
      
        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ ProceduralScalar::toJson(allocator), allocator,
                "type", "code"
            };
        }

        virtual double operator()(Vec3d p) const override {
            return _fn(p);
        }
    };

    class ProceduralVectorCode : public ProceduralVector {
        std::function<Vec3d(Vec3d)> _fn;
    public:
        ProceduralVectorCode(std::function<Vec3d(Vec3d)> fn = nullptr) : _fn(fn) {}

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ ProceduralVector::toJson(allocator), allocator,
                "type", "code"
            };
        }

        virtual Vec3d operator()(Vec3d p) const override {
            return _fn(p);
        }
    };

    class ProceduralSdf : public ProceduralScalar {
        SdfFunctions::Function _fn;
    public:
        ProceduralSdf(SdfFunctions::Function fn = SdfFunctions::Function::Knob) : _fn(fn) {}

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ ProceduralScalar::toJson(allocator), allocator,
                "type", "sdf",
                "function", SdfFunctions::functionToString(_fn)
            };
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            ProceduralScalar::fromJson(value, scene);
            std::string fnString;
            value.getField("function", fnString);
            _fn = SdfFunctions::stringToFunction(fnString);
        }

        virtual double operator()(Vec3d p) const override {
            int mat;
            return SdfFunctions::eval(_fn, vec_conv<Vec3f>(p), mat);
        }
    };

    class ProceduralNoise: public ProceduralScalar {
        double _min = 1., _max = 500.;
        enum class NoiseType {
            BottomTop,
            LeftRight
        };

        NoiseType type = NoiseType::BottomTop;

        static std::string noiseTypeToString(NoiseType v) {
            switch (v) {
            case NoiseType::BottomTop: return "bottom_top";
            case NoiseType::LeftRight: return "left_right";
            }
        }

        static NoiseType stringToNoiseType(std::string v) {
            if (v == "bottom_top")
                return NoiseType::BottomTop;
            else if (v == "left_right")
                return NoiseType::LeftRight;

            FAIL("Invalid noise typ function: '%s'", v);
        }

    public:
        ProceduralNoise() {}

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ ProceduralScalar::toJson(allocator), allocator,
                "type", "noise",
                "noise", noiseTypeToString(type)
            };
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            ProceduralScalar::fromJson(value, scene);
            std::string noiseString = noiseTypeToString(type);
            value.getField("noise", noiseString);
            type = stringToNoiseType(noiseString);

            value.getField("min", _min);
            value.getField("max", _max);
        }

        virtual double operator()(Vec3d p) const override;
    };


    class ProceduralNonstationaryCovariance : public CovarianceFunction {
    public:

        ProceduralNonstationaryCovariance(
            const std::shared_ptr<StationaryCovariance> stationaryCov = nullptr,
            const std::shared_ptr<ProceduralScalar>  variance = nullptr,
            const std::shared_ptr<ProceduralVector>  ls = nullptr,
            const std::shared_ptr<ProceduralVector>  aniso = nullptr) : _stationaryCov(stationaryCov), _variance(variance), _ls(ls), _anisoField(aniso) { }

        virtual double sample_spectral_density(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            if (!_ls) {
                return _stationaryCov->sample_spectral_density(sampler, p);
            }
            else {
                return _stationaryCov->sample_spectral_density(sampler, p) / ((*_ls)(p)[0]);
            }
        }

        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            if (!_ls) {
                return _stationaryCov->sample_spectral_density_2d(sampler, p);
            }
            else if (!_anisoField) {
                auto spec = _stationaryCov->sample_spectral_density_2d(sampler, p);
                spec = spec.length() * spec.normalized() / ((*_ls)(p)).xy();
                return spec;
            }
            else {
                Eigen::Matrix3d anisoMat = getAnisoRoot(p).inverse();
                auto spec = _stationaryCov->sample_spectral_density_2d(sampler, p);
                spec = spec.length() * mult((Eigen::Matrix2d)anisoMat.block(0,0,2,2), spec.normalized());
                return spec;
            }
        }

        virtual Vec3d sample_spectral_density_3d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            if (!_ls) {
                return _stationaryCov->sample_spectral_density_3d(sampler, p);
            }
            else if (!_anisoField) {
                auto spec = _stationaryCov->sample_spectral_density_3d(sampler, p);
                spec = spec.length() * spec.normalized() / ((*_ls)(p));
                return spec;
            }
            else {
                Eigen::Matrix3d anisoMat = getAnisoRoot(p).inverse();
                auto spec = _stationaryCov->sample_spectral_density_3d(sampler, p);
                spec = spec.length() * mult(anisoMat, spec.normalized());
                return spec;
            }
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            CovarianceFunction::fromJson(value, scene);

            if (auto cov = value["cov"]) {
                _stationaryCov = std::dynamic_pointer_cast<StationaryCovariance>(scene.fetchCovarianceFunction(cov));
            }

            if (auto variance = value["var"]) {
                _variance = scene.fetchProceduralScalar(variance);
            }

            if (auto variance = value["ls"]) {
                _ls = scene.fetchProceduralVector(variance);
            }

            if (auto aniso = value["aniso"]) {
                _anisoField = scene.fetchProceduralVector(aniso);
            }
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            auto obj = JsonObject{ CovarianceFunction::toJson(allocator), allocator,
                "type", "proc_nonstationary",
                "cov", *_stationaryCov,
            };

            if (_variance) {
                obj.add("var", *_variance);
            }

            if (_ls) {
                obj.add("ls", *_ls);
            }

            if (_anisoField) {
                obj.add("aniso", *_anisoField);
            }

            return obj;
        }

        virtual void loadResources() override {
            if (_variance) _variance->loadResources();
            if (_ls) _ls->loadResources();
            if (_anisoField) _anisoField->loadResources();
        }

        virtual std::string id() const {
            return tinyformat::format("pns-%s", _stationaryCov->id());
        }

        double getVariance(Vec3d p) const {
            if (_variance) return (*_variance)(p);
            return 1;
        }

        Eigen::Matrix3d getAnisoRoot(Vec3d p) const {
            if (_anisoField && _ls) {
                Vec3d dir = (*_anisoField)(p).normalized();
                Vec3d ls = (*_ls)(p);
                return compute_ansio<Eigen::Matrix3d>(vec_conv<Eigen::Vector3d>(dir), vec_conv<Eigen::Vector3d>(ls));
            }
            else if (_ls) {
                Vec3d ls = (*_ls)(p);
                Eigen::Matrix3d anisoA = Eigen::Matrix3d::Identity();
                anisoA.diagonal().array() *= vec_conv<Eigen::Array3d>(ls);
                return anisoA;
            }

            return Eigen::Matrix3d::Identity();
        }

        Eigen::Matrix3d getAniso(Vec3d p) const {
            if (_anisoField && _ls) {
                Vec3d dir = (*_anisoField)(p).normalized();
                Vec3d ls = (*_ls)(p);
                return compute_ansio<Eigen::Matrix3d>(vec_conv<Eigen::Vector3d>(dir), vec_conv<Eigen::Vector3d>(ls*ls));
            }
            else if(_ls) {
                Vec3d ls = (*_ls)(p);
                Eigen::Matrix3d anisoA = Eigen::Matrix3d::Identity();
                anisoA.diagonal().array() *= vec_conv<Eigen::Array3d>(ls*ls);
                return anisoA;
            }

            return Eigen::Matrix3d::Identity();
        }

    private:

        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override {
            auto sigmaA = getVariance(from_diff(a));
            auto sigmaB = getVariance(from_diff(b));

            if (!_ls) {
                return sigmaA * sigmaB * _stationaryCov->cov(a, b);
            }

            auto anisoA = getAniso(from_diff(a));
            auto anisoB = getAniso(from_diff(b));

            auto detAnisoA = anisoA.determinant();
            auto detAnisoB = anisoB.determinant();

            auto anisoABavg = 0.5 * (anisoA + anisoB);
            auto detAnisoABavg = (anisoABavg).determinant();

            auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

            auto d = vec_conv<Vec3Diff>(b - a);
            FloatD dsq = d.transpose() * anisoABavg.inverse() * d;
            return sigmaA * sigmaB * ansioFac * _stationaryCov->cov(dsq);
        }

        virtual FloatDD cov(Vec3DD a, Vec3DD b) const override {
            auto sigmaA = getVariance(from_diff(a));
            auto sigmaB = getVariance(from_diff(b));

            if (!_ls) {
                return sigmaA * sigmaB * _stationaryCov->cov(a, b);
            }

            auto anisoA = getAniso(from_diff(a));
            auto anisoB = getAniso(from_diff(b));

            auto detAnisoA = anisoA.determinant();
            auto detAnisoB = anisoB.determinant();

            auto anisoABavg = 0.5 * (anisoA + anisoB);
            auto detAnisoABavg = (anisoABavg).determinant();

            auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

            auto d = vec_conv<Vec3DD>(b - a);
            FloatDD dsq = d.transpose() * anisoABavg.inverse() * d;
            return sigmaA * sigmaB * ansioFac * _stationaryCov->cov(dsq);
        }

        virtual double cov(Vec3d a, Vec3d b) const override {
            auto sigmaA = getVariance(a);
            auto sigmaB = getVariance(b);

            if (!_ls) {
                return sigmaA * sigmaB * _stationaryCov->cov(a, b);
            }

            auto anisoA = getAniso(a);
            auto anisoB = getAniso(b);

            auto detAnisoA = anisoA.determinant();
            auto detAnisoB = anisoB.determinant();

            auto anisoABavg = 0.5 * (anisoA + anisoB);
            auto detAnisoABavg = (anisoABavg).determinant();

            auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

            auto d = vec_conv<Eigen::Vector3d>(b - a);
            auto dsq = d.transpose() * anisoABavg.inverse() * d;
            return sigmaA * sigmaB * ansioFac * _stationaryCov->cov(dsq);

        }

        std::shared_ptr<StationaryCovariance> _stationaryCov;
        std::shared_ptr<ProceduralScalar> _variance;
        std::shared_ptr<ProceduralVector> _ls;
        std::shared_ptr<ProceduralVector> _anisoField;
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
        float _offset = 0;
        float _scale = 1;
    };

    class NeuralNonstationaryCovariance : public CovarianceFunction {
    public:

        NeuralNonstationaryCovariance(
            std::shared_ptr<GPNeuralNetwork> nn = nullptr,
            float offset = 0, float scale = 1) : _nn(nn), _offset(offset), _scale(scale)
        {
        }

        virtual bool requireProjection() const { return true; }
        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

        virtual std::string id() const {
            return tinyformat::format("nn-ns");
        }

    private:
        virtual FloatD cov(Vec3Diff a, Vec3Diff b) const override;
        virtual FloatDD cov(Vec3DD a, Vec3DD b) const override;
        virtual double cov(Vec3d a, Vec3d b) const override;

        PathPtr _path;

        Mat4f _configTransform;
        Mat4f _invConfigTransform;

        std::shared_ptr<GPNeuralNetwork> _nn;
        float _offset = 0;
        float _scale = 1;
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

        virtual double sample_spectral_density(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            return abs(rand_normal_2(sampler).x() / _l);
        }

        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            double rad = (sqrt(2) * sqrt(-log(sampler.next1D()))) / _l;
            double angle = sampler.next1D() * 2 * PI;
            return Vec2d(sin(angle), cos(angle)) *  rad;
        }

        virtual Vec3d sample_spectral_density_3d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            Vec3d normal = vec_conv<Vec3d>(sample_standard_normal(3, sampler));
            double rad = sqrt(2) * 1 / _l * normal.length();
            return vec_conv<Vec3d>(SampleWarp::uniformSphere(sampler.next2D())) * rad;
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

        virtual double sample_spectral_density(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            double tau = rand_gamma(_a, 1 / (_l * _l), sampler);
            double l = 1 / sqrt(tau);
            return abs(rand_normal_2(sampler).x() / l);
        }

        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            double rad = 2 * sqrt(rand_gamma(_a, 0.5 / (_l * _l), sampler)) * sqrt(-log(sampler.next1D()));
            double angle = sampler.next1D() * 2 * PI;
            return Vec2d(sin(angle), cos(angle)) * rad;
        }

        virtual Vec3d sample_spectral_density_3d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            double tau = 2 * sqrt(rand_gamma(_a, 0.5 / (_l * _l), sampler));
            Vec3d normal = vec_conv<Vec3d>(sample_standard_normal(3, sampler));
            double rad = sqrt(2) * 1/tau * normal.length();
            return vec_conv<Vec3d>(SampleWarp::uniformSphere(sampler.next2D())) * rad;
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

        virtual double sample_spectral_density(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            return sqrt(-_v + _v * pow(1 - sampler.next1D(), -1. / _v)) / (sqrt(2) * _l * PI) * sin(PI * sampler.next1D());
        }

        virtual Vec2d sample_spectral_density_2d(PathSampleGenerator& sampler, Vec3d p = Vec3d(0.)) const override {
            double r = sqrt(-_v + _v * pow(1 - sampler.next1D(), -1. / _v)) / (sqrt(2) * _l * PI);
            double angle = sampler.next1D() * 2 * PI;
            return Vec2d(sin(angle), cos(angle)) * r;
        }

    private:
        float _sigma, _v, _l;

        FloatD bessel_k(double n, FloatD x) const {
            FloatD result;
            result[0] = std::cyl_bessel_k(n, x.val());
            result[1] = x[1] * (-std::cyl_bessel_k(n - 1, x.val()) - n / x.val() * std::cyl_bessel_k(n + 1, x.val()));
            return result;
        }

        FloatDD bessel_k(double n, FloatDD x) const {
            FloatDD result;
            result.val.val = std::cyl_bessel_k(n, x.val.val);
            result.val.grad = x.val.val * (-std::cyl_bessel_k(n - 1, x.val.val) - n / x.val.val * std::cyl_bessel_k(n + 1, x.val.val));
            return result;
        }

        virtual FloatD cov(FloatD absq) const override {
            auto r_scl = sqrt(2 * _v * absq) / _l;
            return pow(2, 1 - _v) / std::lgamma(_v) * pow(r_scl, _v) * bessel_k(_v, r_scl);
        }

        virtual FloatDD cov(FloatDD absq) const override {
            auto r_scl = sqrt(2 * _v * absq) / _l;
            return pow(2, 1 - _v) / std::lgamma(_v) * pow(r_scl, _v) * bessel_k(_v, r_scl);
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

        LinearMean(Vec3d ref = Vec3d(0.), Vec3d dir = Vec3d(1., 0., 0.), float scale = 1.0f, float min = -FLT_MAX) :
            _ref(ref), _dir(dir.normalized()), _scale(scale), _min(min) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override {
            MeanFunction::fromJson(value, scene);

            value.getField("reference_point", _ref);
            value.getField("direction", _dir);
            value.getField("scale", _scale);
            value.getField("min", _min);

            _dir.normalize();
        }

        virtual rapidjson::Value toJson(Allocator& allocator) const override {
            return JsonObject{ JsonSerializable::toJson(allocator), allocator,
                "type", "linear",
                "reference_point", _ref,
                "direction", _dir,
                "scale", _scale,
                "min", _min
            };
        }

        virtual Affine<1> mean(const Affine<3>& a) const override {
            return dot(_dir, a - _ref) * _scale;
        }

    private:
        Vec3d _ref;
        Vec3d _dir;
        float _scale;
        float _min = -FLT_MAX;

        virtual double mean(Vec3d a) const override {
            return max((a - _ref).dot(_dir) * _scale, (double)_min);
        }

        virtual Vec3d dmean_da(Vec3d a) const override {
            if ((a - _ref).dot(_dir) * _scale < _min) {
                return Vec3d(0.);
            }
            else {
                return _dir * _scale;
            }
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
        float _offset = 0;
        float _scale = 1;

        virtual double mean(Vec3d a) const override;
        virtual Vec3d dmean_da(Vec3d a) const override;
    };

    class NeuralMean : public MeanFunction {
    public:

        NeuralMean(std::shared_ptr<GPNeuralNetwork> nn = nullptr, float offset = 0, float scale = 1) : _nn(nn), _offset(offset), _scale(scale) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

    private:
        std::shared_ptr<GPNeuralNetwork> _nn;
        float _offset = 0;
        float _scale = 1;
        PathPtr _path;

        Mat4f _configTransform;
        Mat4f _invConfigTransform;

        virtual double mean(Vec3d a) const override;
        virtual Vec3d dmean_da(Vec3d a) const override;
    };

    class ProceduralMean : public MeanFunction {
    public:

        ProceduralMean(std::shared_ptr<ProceduralScalar> f) : _f(f) {}
        ProceduralMean(SdfFunctions::Function fn = SdfFunctions::Function::Knob) {
            _f = std::make_shared<ProceduralSdf>(fn);
        }

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;

    private:
        std::shared_ptr<ProceduralScalar> _f;

        float _min = -FLT_MAX;
        float _scale = 1.f;

        Mat4f _configTransform;
        Mat4f _invConfigTransform;

        virtual double mean(Vec3d a) const override;
        virtual Vec3d dmean_da(Vec3d a) const override;
    };

    class MeshSdfMean : public MeanFunction {
    public:

        MeshSdfMean(PathPtr path = nullptr, bool isSigned = false) : _path(path), _signed(isSigned) {}

        virtual void fromJson(JsonPtr value, const Scene& scene) override;
        virtual rapidjson::Value toJson(Allocator& allocator) const override;
        virtual void loadResources() override;

        Box3d bounds() {
            return _bounds;
        }

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

        Box3d _bounds;
        Mat4f _configTransform;
        Mat4f _invConfigTransform;

        virtual double mean(Vec3d a) const override;
        virtual Vec3d dmean_da(Vec3d a) const override;
    };

}

#endif //GPFUNCTIONS_HPP_