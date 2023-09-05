#include "GaussianProcess.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "io/MeshIO.hpp"

#include "ziggurat_constants.h"
#include "primitives/Triangle.hpp"
#include "primitives/Vertex.hpp"
#include <Eigen/SparseQR>
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>

#include <Spectra/MatOp/SparseGenMatProd.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <boost/math/special_functions/erf.hpp>
#include <Eigen/IterativeLinearSolvers>


namespace Tungsten {


#ifdef SPARSE_COV
//#define SPARSE_SOLVE
#endif


FloatD CovarianceFunction::dcov_da(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA) const {
    Eigen::Array3d zd = Eigen::Array3d::Zero();
    auto covDiff = autodiff::derivatives([&](auto a, auto b) { return cov(a,b); }, autodiff::along(dirA, zd), at(a, b));
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
        "cov",* _stationaryCov,
        "mean", *_mean,
        "aniso", _aniso,
    };
}

void MeanGradNonstationaryCovariance::loadResources() {
    _stationaryCov->loadResources();
}


template <typename Mat, typename Vec>
static inline Mat compute_ansio(const Vec&grad, const Vec& aniso) {
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
    auto anisoA = compute_ansio<Mat3Diff>(
        vec_conv<Vec3Diff>(_mean->dmean_da(vec_conv<Vec3d>(a))), 
        vec_conv<Vec3Diff>(_aniso));
    auto anisoB = compute_ansio<Mat3Diff>(
        vec_conv<Vec3Diff>(_mean->dmean_da(vec_conv<Vec3d>(b))),
        vec_conv<Vec3Diff>(_aniso));

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
    auto anisoA = compute_ansio<Mat3DD>(
        vec_conv<Vec3DD>(_mean->dmean_da(vec_conv<Vec3d>(a))),
        vec_conv<Vec3DD>(_aniso));
    auto anisoB = compute_ansio<Mat3DD>(
        vec_conv<Vec3DD>(_mean->dmean_da(vec_conv<Vec3d>(b))),
        vec_conv<Vec3DD>(_aniso));

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

    return Vec3d(vals[0] - vals[3], vals[1] - vals[4], vals[2] - vals[5]) / (2*eps);
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



void GaussianProcess::fromJson(JsonPtr value, const Scene& scene) {
    JsonSerializable::fromJson(value, scene);

    if (auto mean = value["mean"])
        _mean = scene.fetchMeanFunction(mean);
    if (auto cov = value["covariance"])
        _cov = scene.fetchCovarianceFunction(cov);

    value.getField("max_num_eigenvalues", _maxEigenvaluesN);
    value.getField("covariance_epsilon", _covEps);

    if (auto conditioningDataPath = value["conditioning_data"])
        _conditioningDataPath = scene.fetchResource(conditioningDataPath);

}

rapidjson::Value GaussianProcess::toJson(Allocator& allocator) const {
    return JsonObject{ JsonSerializable::toJson(allocator), allocator,
        "type", "standard",
        "conditioning_data",* _conditioningDataPath,
        "mean", *_mean,
        "covariance", *_cov,
        "max_num_eigenvalues", _maxEigenvaluesN,
        "covariance_epsilon", _covEps
    };
}

void GaussianProcess::loadResources() {
    _mean->loadResources();
    _cov->loadResources();

    std::vector<Vertex> verts;
    std::vector<TriangleI> tris;
    if (_conditioningDataPath && MeshIO::load(*_conditioningDataPath, verts, tris)) {
        for (const auto& v : verts) {
            _globalCondPs.push_back(vec_conv<Vec3d>(v.pos()));
            _globalCondDerivs.push_back(Derivative::None);
            _globalCondDerivDirs.push_back(vec_conv<Vec3d>(v.normal()));
            _globalCondValues.push_back(0);

            _globalCondPs.push_back(vec_conv<Vec3d>(v.pos()));
            _globalCondDerivs.push_back(Derivative::First);
            _globalCondDerivDirs.push_back(vec_conv<Vec3d>(v.normal()));
            _globalCondValues.push_back(1);
        }
    }

}

std::tuple<Eigen::VectorXd, CovMatrix> GaussianProcess::mean_and_cov(
    const Vec3d* points, const Derivative* derivative_types, const Vec3d* ddirs,
    Vec3d deriv_dir, size_t numPts) const {

    Eigen::VectorXd ps_mean(numPts);
    CovMatrix ps_cov(numPts, numPts);

#ifdef SPARSE_COV
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(min(numPts * numPts / 10, (size_t)10000));
#endif

    for (size_t i = 0; i < numPts; i++) {
        const Vec3d& ddir_a = ddirs ? ddirs[i] : deriv_dir;
        ps_mean(i) = (*_mean)(derivative_types[i], points[i], ddir_a);

        for (size_t j = 0; j < numPts; j++) {
            const Vec3d& ddir_b = ddirs ? ddirs[j] : deriv_dir;
            double cov_ij = (*_cov)(derivative_types[i], derivative_types[j], points[i], points[j], ddir_a, ddir_b);

#ifdef SPARSE_COV
            if (i == j || std::abs(cov_ij) > _covEps) {
                tripletList.push_back(Eigen::Triplet<double>(i, j, cov_ij));
            }
#else
            ps_cov(i, j) = cov_ij;
#endif
        }
    }

#ifdef SPARSE_COV
    ps_cov.setFromTriplets(tripletList.begin(), tripletList.end());
    ps_cov.makeCompressed();
#endif

    return { ps_mean, ps_cov };
}

Eigen::VectorXd GaussianProcess::mean(
    const Vec3d* points, const Derivative* derivative_types, const Vec3d* ddirs,
    Vec3d deriv_dir, size_t numPts) const {
    Eigen::VectorXd ps_mean(numPts);
    for (size_t i = 0; i < numPts; i++) {
        const Vec3d& ddir = ddirs ? ddirs[i] : deriv_dir;
        ps_mean(i) = (*_mean)(derivative_types[i], points[i], ddir);
    }
    return ps_mean;
}

CovMatrix GaussianProcess::cov(
    const Vec3d* points_a, const Vec3d* points_b, 
    const Derivative* dtypes_a, const Derivative* dtypes_b,
    const Vec3d* ddirs_a, const Vec3d* ddirs_b,
    Vec3d deriv_dir, size_t numPtsA, size_t numPtsB) const {
    CovMatrix ps_cov(numPtsA, numPtsB);

#ifdef SPARSE_COV
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(min(numPtsA * numPtsB / 10, (size_t)10000));
#endif


    for (size_t i = 0; i < numPtsA; i++) {
        const Vec3d& ddir_a = ddirs_a ? ddirs_a[i] : deriv_dir;
        for (size_t j = 0; j < numPtsB; j++) {
            const Vec3d& ddir_b = ddirs_b ? ddirs_b[j] : deriv_dir;

            double cov_ij = (*_cov)(dtypes_a[i], dtypes_b[j], points_a[i], points_b[j], ddir_a, ddir_b);

#ifdef SPARSE_COV
            if (i == j || std::abs(cov_ij) > _covEps) {
                tripletList.push_back(Eigen::Triplet<double>(i, j, cov_ij));
            }
#else
            ps_cov(i, j) = cov_ij;
#endif
        }
    }

#ifdef SPARSE_COV
    ps_cov.setFromTriplets(tripletList.begin(), tripletList.end());
    ps_cov.makeCompressed();
#endif

    return ps_cov;
}

double GaussianProcess::sample_start_value(Vec3d p, PathSampleGenerator& sampler) const {
    double m = (*_mean)(Derivative::None, p, Vec3d(0.f));
    double sigma = sqrt((*_cov)(Derivative::None, Derivative::None, p, p, Vec3d(0.), Vec3d(0.)));

    return max(0., rand_truncated_normal(m, sigma, 0, sampler));
}


Eigen::MatrixXd GaussianProcess::sample(
    const Vec3d* points, const Derivative* derivative_types, size_t numPts,
    const Vec3d* deriv_dirs,
    const Constraint* constraints, size_t numConstraints,
    Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const {

    if (_globalCondPs.size() == 0) {
        auto [ps_mean, ps_cov] = mean_and_cov(points, derivative_types, deriv_dirs, deriv_dir, numPts);
        return sample_multivariate_normal(ps_mean, ps_cov, constraints, numConstraints, samples, sampler);
    }
    else {

        return sample_cond(points, derivative_types, numPts,
            deriv_dirs,
            _globalCondPs.data(), _globalCondValues.data(), _globalCondDerivs.data(), 0,
            _globalCondDerivDirs.data(),
            constraints, numConstraints,
            deriv_dir, samples, sampler);
    }
}

Eigen::MatrixXd GaussianProcess::sample_cond(
    const Vec3d* points, const Derivative* derivative_types, size_t numPts,
    const Vec3d* deriv_dirs,
    const Vec3d* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
    const Vec3d* cond_deriv_dirs,
    const Constraint* constraints, size_t numConstraints,
    Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const {

    std::vector<Vec3d> cond_ps;
    std::vector<Derivative> cond_derivs;
    std::vector<Vec3d> cond_dds;
    std::vector<double> cond_vs;

    if (_globalCondPs.size() > 0) {
        cond_ps.resize(_globalCondPs.size() + numCondPts);
        cond_derivs.resize(_globalCondDerivs.size() + numCondPts);
        cond_dds.resize(_globalCondDerivDirs.size() + numCondPts);
        cond_vs.resize(_globalCondValues.size() + numCondPts);

        std::copy(cond_points, cond_points + numCondPts, cond_ps.begin());
        std::copy(_globalCondPs.begin(), _globalCondPs.end(), cond_ps.begin() + numCondPts);

        std::copy(cond_derivative_types, cond_derivative_types + numCondPts, cond_derivs.begin());
        std::copy(_globalCondDerivs.begin(), _globalCondDerivs.end(), cond_derivs.begin() + numCondPts);

        if (cond_deriv_dirs != nullptr) {
            std::copy(cond_deriv_dirs, cond_deriv_dirs + numCondPts, cond_dds.begin());
        }
        else {
            std::fill_n(cond_dds.begin(), numCondPts, deriv_dir);
        }
        std::copy(_globalCondDerivDirs.begin(), _globalCondDerivDirs.end(), cond_dds.begin() + numCondPts);

        std::copy(cond_values, cond_values + numCondPts, cond_vs.begin());
        std::copy(_globalCondValues.begin(), _globalCondValues.end(), cond_vs.begin() + numCondPts);

        cond_points = cond_ps.data();
        cond_derivative_types = cond_derivs.data();
        cond_deriv_dirs = _globalCondDerivDirs.data();
        cond_values = cond_vs.data();

        numCondPts = cond_ps.size();
    }

    if (numCondPts == 0) {
        return sample(points, derivative_types, numPts, deriv_dirs, constraints, numConstraints, deriv_dir, samples, sampler);
    }

    CovMatrix s11 = cov(
        cond_points, cond_points, 
        cond_derivative_types, cond_derivative_types, 
        cond_deriv_dirs, cond_deriv_dirs, 
        deriv_dir, numCondPts, numCondPts);

    CovMatrix s12 = cov(
        cond_points, points, 
        cond_derivative_types, derivative_types, 
        cond_deriv_dirs, deriv_dirs,
        deriv_dir, numCondPts, numPts);

    CovMatrix solved;

#ifdef SPARSE_COV
    /*Eigen::SparseQR<CovMatrix, Eigen::AMDOrdering<int>> solver(s11);
    if (solver.info() != Eigen::ComputationInfo::Success) {
        std::cerr << "Conditioning failed!\n";
    }*/

    //Eigen::ConjugateGradient<CovMatrix, Eigen::Lower | Eigen::Upper> solver;
    //Eigen::BiCGSTAB<CovMatrix> solver;
    
    if (s11.rows() > 16) {
        Eigen::BDCSVD<Eigen::MatrixXd> solver(s11, Eigen::ComputeThinU | Eigen::ComputeThinV);
        solver.setThreshold(0.00001);

        if (solver.info() != Eigen::ComputationInfo::Success) {
            std::cerr << "Conditioning decomposition failed (BDCSVD)!\n";
        }

        Eigen::MatrixXd solvedDense = solver.solve(s12.toDense()).transpose();
        solved = solvedDense.sparseView();
        if (solver.info() != Eigen::ComputationInfo::Success) {
                std::cerr << "Conditioning solving failed (BDCSVD)!\n";
        }
    }
    else {
        Eigen::SimplicialLDLT<CovMatrix> solver;
        solver.compute(s11);
        if (solver.info() != Eigen::ComputationInfo::Success) {
            std::cerr << "Conditioning decomposition failed (LDLT)!\n";
        }

        solved = solver.solve(s12).transpose();
        if (solver.info() != Eigen::ComputationInfo::Success) {
            std::cerr << "Conditioning solving failed (LDLT)!\n";
        }
    }
#else
    Eigen::HouseholderQR<CovMatrix> solver(s11);
    solver.compute(s11);
    if (solver.info() != Eigen::ComputationInfo::Success) {
        std::cerr << "Conditioning decomposition failed!\n";
    }

    CovMatrix solved = solver.solve(s12).transpose();
    if (solver.info() != Eigen::ComputationInfo::Success) {
        std::cerr << "Conditioning solving failed!\n";
    }
#endif

    Eigen::Map<const Eigen::VectorXd> cond_values_view(cond_values, numCondPts);
    Eigen::VectorXd m2 = mean(points, derivative_types, deriv_dirs, deriv_dir, numPts) + (solved * (cond_values_view - mean(cond_points, cond_derivative_types, cond_deriv_dirs, deriv_dir, numCondPts)));

    CovMatrix s22 = cov(
        points, points, 
        derivative_types, derivative_types, 
        deriv_dirs, deriv_dirs,
        deriv_dir, numPts, numPts);

    CovMatrix s2 = s22 - (solved * s12);

    return sample_multivariate_normal(m2, s2, constraints, numConstraints, samples, sampler);
}



// Get me some bits
uint64_t GaussianProcess::vec2uint(Vec2f v) const {
    return ((uint64_t)BitManip::floatBitsToUint(v.x())) | ((uint64_t)BitManip::floatBitsToUint(v.y()) << 32);
}

// From numpy
double GaussianProcess::random_standard_normal(PathSampleGenerator& sampler) const {
    uint64_t r;
    int sign;
    uint64_t rabs;
    int idx;
    double x, xx, yy;
    for (;;) {
        /* r = e3n52sb8 */
        r = vec2uint(sampler.next2D());
        idx = r & 0xff;
        r >>= 8;
        sign = r & 0x1;
        rabs = (r >> 1) & 0x000fffffffffffff;
        x = rabs * wi_double[idx];
        if (sign & 0x1)
            x = -x;
        if (rabs < ki_double[idx])
            return x; /* 99.3% of the time return here */
        if (idx == 0) {
            for (;;) {
                /* Switch to 1.0 - U to avoid log(0.0), see GH 13361 */
                xx = -ziggurat_nor_inv_r * log(1.0 - sampler.next1D());
                yy = -log(1.0 - sampler.next1D());
                if (yy + yy > xx * xx)
                    return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx)
                    : ziggurat_nor_r + xx;
            }
        }
        else {
            if (((fi_double[idx - 1] - fi_double[idx]) * sampler.next1D() +
                fi_double[idx]) < exp(-0.5 * x * x))
                return x;
        }
    }
}

double GaussianProcess::noIntersectBound(Vec3d p, double q) const
{
    double stddev = sqrt((*_cov)(Derivative::None, Derivative::None, p, p, Vec3d(0.), Vec3d(0.)));
    return stddev * sqrt(2.) * boost::math::erf_inv(2 * q - 1);
}

double GaussianProcess::goodStepsize(Vec3d p, double targetCov) const
{
    targetCov *= (*_cov)(Derivative::None, Derivative::None, p, p, Vec3d(0.), Vec3d(0.));
    double stepsize = 10.;
    double cov = (*_cov)(Derivative::None, Derivative::None, p, p + Vec3d(stepsize, 0., 0.), Vec3d(0.), Vec3d(0.));

    while (std::abs(cov - targetCov) > 0.0000001) {
        if (cov > targetCov) {
            stepsize *= 1.5;
        }
        else {
            stepsize *= 0.5;
        }
        cov = (*_cov)(Derivative::None, Derivative::None, p, p + Vec3d(stepsize, 0., 0.), Vec3d(0.), Vec3d(0.));
    } 

    return stepsize;
}

// Box muller transform
Vec2d GaussianProcess::rand_normal_2(PathSampleGenerator& sampler) const {
    double u1 = sampler.next1D();
    double u2 = sampler.next1D();

    double r = sqrt(-2 * log(1. - u1));
    double x = cos(2 * PI * u2);
    double y = sin(2 * PI * u2);
    double z1 = r * x;
    double z2 = r * y;

    return Vec2d(z1, z2);

}

double GaussianProcess::rand_truncated_normal(double mean, double sigma, double a, PathSampleGenerator& sampler) const {
    if (abs(a - mean) < 0.000001) {
        return abs(mean + sigma * rand_normal_2(sampler).x());
    }

    if (a < mean) {
        while (true) {
            double x = mean + sigma * rand_normal_2(sampler).x();
            if (x >= a) {
                return x;
            }
        }
    }

    double a_bar = (a - mean) / sigma;
    double x_bar;

    while (true) {
        double u = sampler.next1D();
        x_bar = sqrt(a_bar * a_bar - 2 * log(1 - u));
        double v = sampler.next1D();
        
        if (v < x_bar / a_bar) {
            break;
        }
    }

    return sigma * x_bar + mean;
}

Eigen::MatrixXd GaussianProcess::sample_multivariate_normal(
    const Eigen::VectorXd& mean, const CovMatrix& cov,
    const Constraint* constraints, int numConstraints,
    int samples, PathSampleGenerator& sampler) const {

    Eigen::MatrixXd normTransform;

#ifndef SPARSE_SOLVE
    // Use the Cholesky decomposition to transform the standard normal vector to the desired multivariate normal distribution

#ifdef SPARSE_COV
    Eigen::SimplicialLLT<CovMatrix> chol(cov);
#else
    Eigen::LLT<Eigen::MatrixXd> chol(cov);
#endif

    // We can only use the cholesky decomposition if 
      // the covariance matrix is symmetric, pos-definite.
      // But a covariance matrix might be pos-semi-definite.
      // In that case, we'll go to an EigenSolver
    if (chol.info() == Eigen::Success) {
        // Use cholesky solver
        normTransform = chol.matrixL();
    }
    else 
#endif
    {

#ifdef SPARSE_SOLVE
        Spectra::SparseGenMatProd<float> op(cov + reg);
        Spectra::SymEigsSolver<Spectra::SparseGenMatProd<float>> eigs(
            op, 
            min(size_t(cov.rows()-2), _maxEigenvaluesN), 
            min(size_t(cov.rows()-1), _maxEigenvaluesN * 2)
        );

        // Initialize and compute
        eigs.init();
        int nconv = eigs.compute(Spectra::SortRule::LargestMagn);
#else
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(cov);
#endif

        if (eigs.info() != Eigen::ComputationInfo::Success) {
            std::cerr << "Matrix square root failed!\n";
        }

        normTransform = eigs.eigenvectors() 
            * eigs.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    }

    // Generate a vector of standard normal variates with the same dimension as the mean
    Eigen::VectorXd z = Eigen::VectorXd(mean.size());
    Eigen::MatrixXd sample(mean.size(), samples);

    int numTries = 0;
    for (int j = 0; j < samples; /*only advance sample idx if the sample passes all constraints*/) {

        numTries++;

        // We're always getting two samples, so make use of that
        for (int i = 0; i < mean.size() / 2; i++) {
            Vec2d norm_samp = rand_normal_2(sampler); // { (float)random_standard_normal(sampler), (float)random_standard_normal(sampler) };
            z(i*2) = norm_samp.x();
            z(i*2 + 1) = norm_samp.y();
        }

        // Fill up the last one for an uneven number of samples
        if (mean.size() % 2) {
            Vec2d norm_samp = rand_normal_2(sampler);
            z(mean.size()-1) = norm_samp.x();
        }

        Eigen::VectorXd currSample = mean + normTransform * z;

        // Check constraints
        bool passedConstraints = true;
        for (int cIdx = 0; cIdx < numConstraints; cIdx++) {
            const Constraint& con = constraints[cIdx];

            for (int i = con.startIdx; i <= con.endIdx; i++) {
                if (currSample(i) < con.minV || currSample(i) > con.maxV) {
                    //currSample *= -1;
                    passedConstraints = false;
                    break;
                }
            }

            if (!passedConstraints) {
                break;
            }
        }

        //sample.col(j) = currSample;
        //j++;

        if (passedConstraints || numTries > 100000) {
            if (numTries > 100000) {
                std::cout << "Constraint not satisfied. " << mean(0) << "\n";
            }
            sample.col(j) = currSample;
            j++;
            numTries = 0;
        }
    }

    return sample;
}
}