#include "GaussianProcess.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "io/MeshIO.hpp"

#include "ziggurat_constants.h"
#include "primitives/Triangle.hpp"
#include "primitives/Vertex.hpp"
#include <Eigen/SparseQR>
#include <Eigen/Core>

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

        for (size_t j = 0; j <= i; j++) {
            const Vec3d& ddir_b = ddirs ? ddirs[j] : deriv_dir;
            double cov_ij = (*_cov)(derivative_types[i], derivative_types[j], points[i], points[j], ddir_a, ddir_b);

#ifdef SPARSE_COV
            if (i == j || std::abs(cov_ij) > _covEps) {
                tripletList.push_back(Eigen::Triplet<double>(i, j, cov_ij));
            }
#else
            ps_cov(i, j) = ps_cov(j, i) = cov_ij;
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

CovMatrix GaussianProcess::cov_sym(
    const Vec3d* points_a,
    const Derivative* dtypes_a,
    const Vec3d* ddirs_a,
    Vec3d deriv_dir, size_t numPtsA) const {
    CovMatrix ps_cov(numPtsA, numPtsA);

#ifdef SPARSE_COV
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(min(numPtsA * numPtsB / 10, (size_t)10000));
#endif


    for (size_t i = 0; i < numPtsA; i++) {
        const Vec3d& ddir_a = ddirs_a ? ddirs_a[i] : deriv_dir;
        for (size_t j = 0; j <= i; j++) {
            const Vec3d& ddir_b = ddirs_a ? ddirs_a[j] : deriv_dir;

            double cov_ij = (*_cov)(dtypes_a[i], dtypes_a[j], points_a[i], points_a[j], ddir_a, ddir_b);

#ifdef SPARSE_COV
            if (i == j || std::abs(cov_ij) > _covEps) {
                tripletList.push_back(Eigen::Triplet<double>(i, j, cov_ij));
                tripletList.push_back(Eigen::Triplet<double>(j, i, cov_ij));
            }
#else
            ps_cov(j, i) = ps_cov(i, j) = cov_ij;
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
        auto mvn = MultivariateNormalDistribution(ps_mean, ps_cov);
        return mvn.sample(constraints, numConstraints, samples, sampler);
    }
    else {
        auto mvn = create_mvn_cond(points, derivative_types, numPts,
            deriv_dirs,
            _globalCondPs.data(), _globalCondValues.data(), _globalCondDerivs.data(), 0,
            _globalCondDerivDirs.data(),
            deriv_dir);
        return mvn.sample(constraints, numConstraints, samples, sampler);
    }
}

Eigen::MatrixXd GaussianProcess::sample_cond(
    const Vec3d* points, const Derivative* derivative_types, size_t numPts,
    const Vec3d* deriv_dirs,
    const Vec3d* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
    const Vec3d* cond_deriv_dirs,
    const Constraint* constraints, size_t numConstraints,
    Vec3d deriv_dir, int samples, PathSampleGenerator& sampler) const {

    auto mvn = create_mvn_cond(points, derivative_types, numPts, deriv_dirs,
        cond_points, cond_values, cond_derivative_types, numCondPts, cond_deriv_dirs,
        deriv_dir);

    return mvn.sample(constraints, numConstraints, samples, sampler);
}


double GaussianProcess::eval(
    const Vec3d* points, const double* values, const Derivative* derivative_types, size_t numPts,
    const Vec3d* ddirs,
    Vec3d deriv_dir) const {

    Eigen::Map<const Eigen::VectorXd> eval_values_View(values, numPts);
    if (_globalCondPs.size() == 0) {
        auto [ps_mean, ps_cov] = mean_and_cov(points, derivative_types, ddirs, deriv_dir, numPts);
        auto mvn = MultivariateNormalDistribution(ps_mean, ps_cov);
        return mvn.eval(eval_values_View);
    }
    else {
        auto mvn = create_mvn_cond(points, derivative_types, numPts,
            ddirs,
            _globalCondPs.data(), _globalCondValues.data(), _globalCondDerivs.data(), 0,
            _globalCondDerivDirs.data(),
            deriv_dir);
        return mvn.eval(eval_values_View);
    }
}

MultivariateNormalDistribution GaussianProcess::create_mvn_cond(
    const Vec3d* points, const Derivative* derivative_types, size_t numPts,
    const Vec3d* ddirs,
    const Vec3d* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
    const Vec3d* cond_ddirs,
    Vec3d deriv_dir) const {

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

        if (cond_ddirs != nullptr) {
            std::copy(cond_ddirs, cond_ddirs + numCondPts, cond_dds.begin());
        }
        else {
            std::fill_n(cond_dds.begin(), numCondPts, deriv_dir);
        }
        std::copy(_globalCondDerivDirs.begin(), _globalCondDerivDirs.end(), cond_dds.begin() + numCondPts);

        std::copy(cond_values, cond_values + numCondPts, cond_vs.begin());
        std::copy(_globalCondValues.begin(), _globalCondValues.end(), cond_vs.begin() + numCondPts);

        cond_points = cond_ps.data();
        cond_derivative_types = cond_derivs.data();
        cond_ddirs = _globalCondDerivDirs.data();
        cond_values = cond_vs.data();

        numCondPts = cond_ps.size();
    }

    if (numCondPts == 0) {
        auto [ps_mean, ps_cov] = mean_and_cov(points, derivative_types, ddirs, deriv_dir, numPts);
        return MultivariateNormalDistribution(ps_mean, ps_cov);
    }

    CovMatrix s11 = cov_sym(
        cond_points,
        cond_derivative_types,
        cond_ddirs,
        deriv_dir, numCondPts);

    CovMatrix s12 = cov(
        cond_points, points,
        cond_derivative_types, derivative_types,
        cond_ddirs, ddirs,
        deriv_dir, numCondPts, numPts);

    CovMatrix solved;

    bool succesfullSolve = false;
    if (true || s11.rows() <= 64) {
#ifdef SPARSE_COV
        Eigen::SimplicialLLT<CovMatrix> solver(s11);
#else
        Eigen::LLT<CovMatrix> solver(s11.triangularView<Eigen::Lower>());
#endif
        if (solver.info() == Eigen::ComputationInfo::Success) {
            solved = solver.solve(s12).transpose();
            if (solver.info() == Eigen::ComputationInfo::Success) {
                succesfullSolve = true;
            }
            else {
                std::cerr << "Conditioning solving failed (LDLT)!\n";
            }
        }
    }

    if (!succesfullSolve) {
        Eigen::BDCSVD<Eigen::MatrixXd> solver(s11.triangularView<Eigen::Lower>(), Eigen::ComputeThinU | Eigen::ComputeThinV);
        solver.setThreshold(0.);

        if (solver.info() != Eigen::ComputationInfo::Success) {
            std::cerr << "Conditioning decomposition failed (BDCSVD)!\n";
        }

#ifdef SPARSE_COV
        Eigen::MatrixXd solvedDense = solver.solve(s12.toDense()).transpose();
        solved = solvedDense.sparseView();
#else
        solved = solver.solve(s12).transpose();
#endif
        if (solver.info() != Eigen::ComputationInfo::Success) {
            std::cerr << "Conditioning solving failed (BDCSVD)!\n";
        }
    }

    Eigen::Map<const Eigen::VectorXd> cond_values_view(cond_values, numCondPts);
    Eigen::VectorXd m2 = mean(points, derivative_types, ddirs, deriv_dir, numPts) + (solved * (cond_values_view - mean(cond_points, cond_derivative_types, cond_ddirs, deriv_dir, numCondPts)));

    CovMatrix s22 = cov_sym(
        points,
        derivative_types,
        ddirs,
        deriv_dir, numPts);

    CovMatrix covAdjust = (solved * s12);

    CovMatrix s2 = s22 - covAdjust;

    return MultivariateNormalDistribution(m2, s2);
}

double GaussianProcess::eval_cond(
    const Vec3d* points, const double* values, const Derivative* derivative_types, size_t numPts,
    const Vec3d* ddirs,
    const Vec3d* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
    const Vec3d* cond_ddirs,
    Vec3d deriv_dir) const {

    Eigen::Map<const Eigen::VectorXd> eval_values_View(values, numPts);
    auto mvn = create_mvn_cond(points, derivative_types, numPts, ddirs, 
        cond_points, cond_values, cond_derivative_types, numCondPts, cond_ddirs,
        deriv_dir);

    return mvn.eval(eval_values_View);
}

double GaussianProcess::noIntersectBound(Vec3d p, double q) const
{
    double stddev = sqrt((*_cov)(Derivative::None, Derivative::None, p, p, Vec3d(0.), Vec3d(0.)));
    return stddev * sqrt(2.) * boost::math::erf_inv(2 * q - 1);
}

double GaussianProcess::goodStepsize(Vec3d p, double targetCov, Vec3d rd) const
{
    targetCov *= (*_cov)(Derivative::None, Derivative::None, p, p, Vec3d(0.), Vec3d(0.));

    double stepsize_lb = 0;
    double stepsize_ub = 2;

    double stepsize_avg = (stepsize_lb + stepsize_ub) * 0.5;
    double cov = (*_cov)(Derivative::None, Derivative::None, p, p + rd * stepsize_avg, Vec3d(0.), Vec3d(0.));

    size_t it = 0;
    while (std::abs(cov - targetCov) > 0.00000000001 && it++ < 100) {
        stepsize_avg = (stepsize_lb + stepsize_ub) * 0.5;
        if (cov > targetCov) {
            stepsize_lb = stepsize_avg;
        }
        else {
            stepsize_ub = stepsize_avg;
        }
        cov = (*_cov)(Derivative::None, Derivative::None, p, p + rd * stepsize_avg, Vec3d(0.), Vec3d(0.));
    } 

    return stepsize_avg;
}


}