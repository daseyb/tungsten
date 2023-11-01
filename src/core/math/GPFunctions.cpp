#include "GPFunctions.hpp"
#include <math/Mat4f.hpp>
#include <math/MathUtil.hpp>
#include <math/Angle.hpp>

#include "io/Scene.hpp"
#include "io/MeshIO.hpp"

#include "primitives/Triangle.hpp"
#include "primitives/Vertex.hpp"
#include <Eigen/SparseQR>
#include <Eigen/Core>

#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <boost/math/special_functions/erf.hpp>
#include <Eigen/IterativeLinearSolvers>
#include <igl/signed_distance.h>

#include <ccomplex>
#include <fftw3.h>
#include <random>

namespace Tungsten {

    void CovarianceFunction::loadResources() {
        if (hasAnalyticSpectralDensity()) return;

        double max_t = 10;
        std::vector<double> covValues(pow(2, 12));

        size_t i;
        covValues[0] = cov(Vec3d(0.), Vec3d(0.));
        for (i = 1; i < covValues.size(); i++) {
            double t = double(i) / covValues.size() * max_t;
            covValues[i] = cov(Vec3d(0.), Vec3d(t, 0., 0.)) / covValues[0];
        }

        auto dt = max_t / covValues.size();
        auto n = covValues.size();
        auto nfft = 2 * n - 2;
        auto nf = nfft / 2;

        // This is based on the pywafo tospecdata function: https://github.com/wafo-project/pywafo/blob/master/src/wafo/covariance/core.py#L163
        std::vector<double> acf;
        acf.insert(acf.end(), covValues.begin(), covValues.end());
        acf.insert(acf.end(), nfft - 2 * n + 2, 0.);
        acf.insert(acf.end(), covValues.rbegin() + 2, covValues.rend());

        std::vector<std::complex<double>> spectrumValues(acf.size());
        fftw_plan plan = fftw_plan_dft_r2c_1d(acf.size(), acf.data(), (fftw_complex*)spectrumValues.data(), FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        std::vector<double> r_per;
        std::transform(spectrumValues.begin(), spectrumValues.end(), std::back_inserter(r_per), [](auto spec) { return std::max(spec.real(), 0.); });

        discreteSpectralDensity = std::vector<double>();
        std::transform(r_per.begin(), r_per.begin() + nf + 1, std::back_inserter(discreteSpectralDensity), [dt](auto per) { return std::abs(per) * dt / PI; });
    }

    double CovarianceFunction::spectral_density(double s) const {
        double max_t = 10;
        double dt = max_t / pow(2, 12);
        double max_w = PI / dt;

        double bin_c = s / max_w * discreteSpectralDensity.size();
        size_t bin = clamp(size_t(bin_c), size_t(0), discreteSpectralDensity.size() - 1);
        size_t n_bin = clamp(size_t(bin_c) + 1, size_t(0), discreteSpectralDensity.size() - 1);

        double bin_frac = bin_c - bin;

        return lerp(discreteSpectralDensity[bin], discreteSpectralDensity[n_bin], bin_frac);
    }


    double CovarianceFunction::sample_spectral_density(PathSampleGenerator& sampler) const {
        return 0;
    }

    Vec2d CovarianceFunction::sample_spectral_density_2d(PathSampleGenerator& sampler) const {
        return Vec2d(0.);
    }

    Vec3d CovarianceFunction::sample_spectral_density_3d(PathSampleGenerator& sampler) const {
        return Vec3d(0.);
    }


    FloatD CovarianceFunction::dcov_da(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA) const {
        Eigen::Array3d zd = Eigen::Array3d::Zero();
        auto covDiff = autodiff::derivatives([&](auto a, auto b) { return cov(a, b); }, autodiff::along(dirA, zd), at(a, b));
        return covDiff[1];
    }

    FloatD CovarianceFunction::dcov_db(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirB) const {
        return dcov_da(b, a, dirB);
    }

    FloatDD CovarianceFunction::dcov2_dadb(Vec3DD a, Vec3DD b, Eigen::Array3d dirA, Eigen::Array3d dirB) const {
        Eigen::Matrix3d hess = autodiff::hessian([&](auto a, auto b) { return cov(a, b); }, wrt(a, b), at(a, b)).block(3, 0, 3, 3);
        double res = dirA.transpose().matrix() * hess * dirB.matrix();
        return res;
    }


    void NonstationaryCovariance::fromJson(JsonPtr value, const Scene& scene) {
        CovarianceFunction::fromJson(value, scene);

        if (auto cov = value["cov"]) {
            _stationaryCov = std::dynamic_pointer_cast<StationaryCovariance>(scene.fetchCovarianceFunction(cov));
        }

        if (auto variance = value["grid"]) {
            _variance = scene.fetchGrid(variance);
        }
        else if (auto variance = value["variance"]) {
            _variance = scene.fetchGrid(variance);
        }

        if (_variance) {
            _variance->requestGradient();
        }

        if (auto aniso = value["ansio"]) {
            _aniso = scene.fetchGrid(aniso);
            _aniso->requestGradient();
        }

        value.getField("offset", _offset);
        value.getField("scale", _scale);
    }

    rapidjson::Value NonstationaryCovariance::toJson(Allocator& allocator) const {
        return JsonObject{ JsonSerializable::toJson(allocator), allocator,
            "type", "nonstationary",
            "cov", *_stationaryCov,
            "variance", *_variance,
            "offset", _offset,
            "scale", _scale
        };
    }

    void NonstationaryCovariance::loadResources() {
        CovarianceFunction::loadResources();

        _variance->loadResources();
        _stationaryCov->loadResources();

        if (_aniso) {
            _variance->loadResources();
        }

    }

    FloatD NonstationaryCovariance::sampleGrid(Vec3Diff a) const {
        FloatD result;
        Vec3f ap = vec_conv<Vec3f>(from_diff(a));
        result[0] = _variance->density(ap);

        /**/
        float eps = 0.001f;
        float vals[] = {
            _variance->density(ap + Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap + Vec3f(0.f, eps, 0.f)),
            _variance->density(ap + Vec3f(0.f, 0.f, eps)),
            _variance->density(ap - Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap - Vec3f(0.f, eps, 0.f)),
            _variance->density(ap - Vec3f(0.f, 0.f, eps))
        };
        auto grad = Vec3d(vals[0] - vals[3], vals[1] - vals[4], vals[2] - vals[5]) / (2 * eps);

        result[1] = grad.dot({ (float)a.x()[1], (float)a.y()[1] , (float)a.z()[1] });
        result[2] = 0; // linear interp
        return result;
    }

    FloatDD NonstationaryCovariance::sampleGrid(Vec3DD a) const {
        FloatDD result;
        Vec3f ap = vec_conv<Vec3f>(from_diff(a));
        result.val = _variance->density(ap);

        /**/
        float eps = 0.001f;
        float vals[] = {
            _variance->density(ap + Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap + Vec3f(0.f, eps, 0.f)),
            _variance->density(ap + Vec3f(0.f, 0.f, eps)),
            _variance->density(ap - Vec3f(eps, 0.f, 0.f)),
            _variance->density(ap - Vec3f(0.f, eps, 0.f)),
            _variance->density(ap - Vec3f(0.f, 0.f, eps))
        };
        auto grad = Vec3d(vals[0] - vals[3], vals[1] - vals[4], vals[2] - vals[5]) / (2 * eps);

        result.grad.val = grad.dot({ a.x().grad.val, a.y().grad.val , a.z().grad.val });
        result.grad.grad = 0; // linear interp
        return result;
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

    FloatD NonstationaryCovariance::cov(Vec3Diff a, Vec3Diff b) const {

        FloatD sigmaA = (sampleGrid(mult(_variance->invNaturalTransform(), a)) + _offset) * _scale;
        FloatD sigmaB = (sampleGrid(mult(_variance->invNaturalTransform(), b)) + _offset) * _scale;
        //return sqrt(sigmaA) * sqrt(sigmaB) * _stationaryCov->cov(a, b);

        Mat3Diff anisoA = Mat3Diff::Identity();
        Mat3Diff anisoB = Mat3Diff::Identity();

        FloatD detAnisoA = anisoA.determinant();
        FloatD detAnisoB = anisoB.determinant();

        Mat3Diff anisoABavg = 0.5 * (anisoA + anisoB);
        FloatD detAnisoABavg = anisoABavg.determinant();

        FloatD ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        Vec3Diff d = b - a;
        FloatD dsq = d.transpose() * anisoABavg.inverse() * d;
        return sqrt(sigmaA) * sqrt(sigmaB) * ansioFac * _stationaryCov->cov(dsq);
    }

    FloatDD NonstationaryCovariance::cov(Vec3DD a, Vec3DD b) const {

        auto sigmaA = (sampleGrid(mult(_variance->invNaturalTransform(), a)) + _offset) * _scale;
        auto sigmaB = (sampleGrid(mult(_variance->invNaturalTransform(), b)) + _offset) * _scale;
        //return sqrt(sigmaA) * sqrt(sigmaB) * _stationaryCov->cov(a, b);

        auto anisoA = Mat3DD::Identity();
        auto anisoB = Mat3DD::Identity();

        auto detAnisoA = anisoA.determinant();
        auto detAnisoB = anisoB.determinant();

        auto anisoABavg = 0.5 * (anisoA + anisoB);
        auto detAnisoABavg = anisoABavg.determinant();

        auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        auto d = b - a;
        auto dsq = d.transpose() * anisoABavg.inverse() * d;
        return sqrt(sigmaA) * sqrt(sigmaB) * ansioFac * _stationaryCov->cov(dsq);
    }

    double NonstationaryCovariance::cov(Vec3d a, Vec3d b) const {
        double sigmaA = (_variance->density(_variance->invNaturalTransform() * vec_conv<Vec3f>(a)) + _offset) * _scale;
        double sigmaB = (_variance->density(_variance->invNaturalTransform() * vec_conv<Vec3f>(b)) + _offset) * _scale;
        //return sqrt(sigmaA) * sqrt(sigmaB) * _stationaryCov->cov(a, b);

        Eigen::Matrix3f anisoA = Eigen::Matrix3f::Identity();
        Eigen::Matrix3f anisoB = Eigen::Matrix3f::Identity();

        double detAnisoA = anisoA.determinant();
        double detAnisoB = anisoB.determinant();

        Eigen::Matrix3f anisoABavg = 0.5 * (anisoA + anisoB);
        double detAnisoABavg = anisoABavg.determinant();

        double ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        Eigen::Vector3f d = vec_conv<Eigen::Vector3f>(b - a);
        double dsq = d.transpose() * anisoABavg.inverse() * d;
        return sqrt(sigmaA) * sqrt(sigmaB) * ansioFac * _stationaryCov->cov(dsq);
    }


    void MeanGradNonstationaryCovariance::fromJson(JsonPtr value, const Scene& scene) {
        CovarianceFunction::fromJson(value, scene);

        if (auto cov = value["cov"]) {
            _stationaryCov = std::dynamic_pointer_cast<StationaryCovariance>(scene.fetchCovarianceFunction(cov));
        }

        if (auto mean = value["mean"]) {
            _mean = std::dynamic_pointer_cast<MeanFunction>(scene.fetchMeanFunction(mean));
        }

        value.getField("aniso", _aniso);
    }

    rapidjson::Value MeanGradNonstationaryCovariance::toJson(Allocator& allocator) const {
        return JsonObject{ JsonSerializable::toJson(allocator), allocator,
            "type", "mg-nonstationary",
            "cov",*_stationaryCov,
            "mean", *_mean,
            "aniso", _aniso,
        };
    }

    void MeanGradNonstationaryCovariance::loadResources() {
        _stationaryCov->loadResources();
    }


    template <typename Mat, typename Vec>
    static inline Mat compute_ansio(const Vec& grad, const Vec& aniso) {
        TangentFrameD<Mat, Vec> tf(grad);

        auto vmat = tf.toMatrix();
        Mat smat = Mat::Identity();
        smat.diagonal() = aniso;

        return vmat * smat * vmat.transpose();
    }

    Eigen::Matrix3d MeanGradNonstationaryCovariance::localAniso(Vec3d p) const {
        return compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(p).normalized()),
            vec_conv<Eigen::Vector3d>(_aniso));
    }

    FloatD MeanGradNonstationaryCovariance::cov(Vec3Diff a, Vec3Diff b) const {
        Eigen::Matrix3d anisoA = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(a))),
            vec_conv<Eigen::Vector3d>(_aniso));
        Eigen::Matrix3d anisoB = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(b))),
            vec_conv<Eigen::Vector3d>(_aniso));

        auto detAnisoA = anisoA.determinant();
        auto detAnisoB = anisoB.determinant();

        auto anisoABavg = 0.5 * (anisoA + anisoB);
        auto detAnisoABavg = anisoABavg.determinant();

        auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        auto d = b - a;
        auto dsq = d.transpose() * anisoABavg.inverse() * d;
        return ansioFac * _stationaryCov->cov(dsq);
    }

    FloatDD MeanGradNonstationaryCovariance::cov(Vec3DD a, Vec3DD b) const {
        Eigen::Matrix3d anisoA = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(a))),
            vec_conv<Eigen::Vector3d>(_aniso));
        Eigen::Matrix3d anisoB = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(vec_conv<Vec3d>(b))),
            vec_conv<Eigen::Vector3d>(_aniso));

        auto detAnisoA = anisoA.determinant();
        auto detAnisoB = anisoB.determinant();

        auto anisoABavg = 0.5 * (anisoA + anisoB);
        auto detAnisoABavg = anisoABavg.determinant();

        auto ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        auto d = b - a;
        auto dsq = d.transpose() * anisoABavg.inverse() * d;
        return ansioFac * _stationaryCov->cov(dsq);
    }

    double MeanGradNonstationaryCovariance::cov(Vec3d a, Vec3d b) const {
        auto anisoA = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(a)),
            vec_conv<Eigen::Vector3d>(_aniso));
        auto anisoB = compute_ansio<Eigen::Matrix3d>(
            vec_conv<Eigen::Vector3d>(_mean->dmean_da(b)),
            vec_conv<Eigen::Vector3d>(_aniso));

        auto detAnisoA = anisoA.determinant();
        auto detAnisoB = anisoB.determinant();

        auto anisoABavg = 0.5 * (anisoA + anisoB);
        auto detAnisoABavg = anisoABavg.determinant();

        float ansioFac = pow(detAnisoA, 0.25f) * pow(detAnisoB, 0.25f) / sqrt(detAnisoABavg);

        Eigen::Vector3d d = vec_conv<Eigen::Vector3d>(b - a);
        double dsq = d.transpose() * anisoABavg.inverse() * d;
        return ansioFac * _stationaryCov->cov(dsq);
    }

    void TabulatedMean::fromJson(JsonPtr value, const Scene& scene) {
        MeanFunction::fromJson(value, scene);

        if (auto grid = value["grid"]) {
            _grid = scene.fetchGrid(grid);
            _grid->requestSDF();
            _grid->requestGradient();
        }

        value.getField("offset", _offset);
        value.getField("scale", _scale);
    }

    rapidjson::Value TabulatedMean::toJson(Allocator& allocator) const {
        return JsonObject{ JsonSerializable::toJson(allocator), allocator,
            "type", "tabulated",
            "grid", *_grid,
            "offset", _offset,
            "scale", _scale
        };
    }


    double TabulatedMean::mean(Vec3d a) const {
        Vec3f p = _grid->invNaturalTransform() * vec_conv<Vec3f>(a);
        return (_grid->density(p) + _offset) * _scale;
    }

    Vec3d TabulatedMean::dmean_da(Vec3d a) const {
        double eps = 0.001;
        double vals[] = {
            mean(a + Vec3d(eps, 0.f, 0.f)),
            mean(a + Vec3d(0.f, eps, 0.f)),
            mean(a + Vec3d(0.f, 0.f, eps)),
            mean(a - Vec3d(eps, 0.f, 0.f)),
            mean(a - Vec3d(0.f, eps, 0.f)),
            mean(a - Vec3d(0.f, 0.f, eps))
        };

        return Vec3d(vals[0] - vals[3], vals[1] - vals[4], vals[2] - vals[5]) / (2 * eps);
        /*Vec3d p = _grid->invNaturalTransform() * a;
        return _scale* _grid->naturalTransform().transformVector(_grid->gradient(p));*/
    }

    void TabulatedMean::loadResources() {
        _grid->loadResources();
    }


    void MeshSdfMean::fromJson(JsonPtr value, const Scene& scene) {
        MeanFunction::fromJson(value, scene);
        if (auto path = value["file"]) _path = scene.fetchResource(path);
        value.getField("transform", _configTransform);
        value.getField("signed", _signed);
        _invConfigTransform = _configTransform.invert();

    }

    rapidjson::Value MeshSdfMean::toJson(Allocator& allocator) const {
        return JsonObject{ JsonSerializable::toJson(allocator), allocator,
            "type", "mesh",
            "file", *_path,
            "transform", _configTransform,
            "signed", _signed,
        };
    }


    double MeshSdfMean::mean(Vec3d a) const {
        // perform a closest point query
        Eigen::MatrixXd V_vis(1, 3);
        V_vis(0, 0) = a.x();
        V_vis(1, 0) = a.y();
        V_vis(2, 0) = a.z();

        Eigen::VectorXd S_vis;
        igl::signed_distance_fast_winding_number(V_vis, V, F, tree, fwn_bvh, S_vis);

        return (float)S_vis(0);
    }

    Vec3d MeshSdfMean::dmean_da(Vec3d a) const {
        double eps = 0.001f;
        double vals[] = {
            mean(a),
            mean(a + Vec3d(eps, 0.f, 0.f)),
            mean(a + Vec3d(0.f, eps, 0.f)),
            mean(a + Vec3d(0.f, 0.f, eps))
        };

        return Vec3d(vals[1] - vals[0], vals[2] - vals[0], vals[3] - vals[0]) / eps;
    }

    void MeshSdfMean::loadResources() {

        std::vector<Vertex> _verts;
        std::vector<TriangleI> _tris;

        if (_path && MeshIO::load(*_path, _verts, _tris)) {

            V.resize(_verts.size(), 3);

            for (int i = 0; i < _verts.size(); i++) {
                Vec3f tpos = _configTransform * _verts[i].pos();
                V(i, 0) = tpos.x();
                V(i, 1) = tpos.y();
                V(i, 2) = tpos.z();

                Vec3f tnorm = _configTransform.transformVector(_verts[i].normal());
            }

            F.resize(_tris.size(), 3);

            // specify the triangle indices
            for (int i = 0; i < _tris.size(); i++) {
                F(i, 0) = _tris[i].v0;
                F(i, 1) = _tris[i].v1;
                F(i, 2) = _tris[i].v2;
            }

            tree.init(V, F);
            igl::per_face_normals(V, F, FN);
            igl::per_vertex_normals(
                V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, FN, VN);
            igl::per_edge_normals(
                V, F, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, FN, EN, E, EMAP);

            igl::fast_winding_number(V, F, 2, fwn_bvh);
        }
    }

   
}