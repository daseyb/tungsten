#include "GaussianProcess.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "io/MeshIO.hpp"

#include "ziggurat_constants.h"
#include "primitives/Triangle.hpp"
#include "primitives/Vertex.hpp"
#include <fcpw/fcpw.h>
#include <Eigen/SparseQR>
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>

#include <Spectra/MatOp/SparseGenMatProd.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>


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

FloatD CovarianceFunction::dcov2_dadb(Vec3Diff a, Vec3Diff b, Eigen::Array3d dirA, Eigen::Array3d dirB) const {
    auto covDiff = autodiff::derivatives([&](auto a, auto b) { return cov(a, b); }, autodiff::along(Eigen::Array3d(-dirA*0.5f), Eigen::Array3d(dirB*0.5f)), at(a, b));
    return -covDiff[2];
}


void NonstationaryCovariance::fromJson(JsonPtr value, const Scene& scene) {
    CovarianceFunction::fromJson(value, scene);

    if (auto cov = value["cov"]) {
        _stationaryCov = scene.fetchCovarianceFunction(cov);
    }

    if (auto grid = value["grid"]) {
        _grid = scene.fetchGrid(grid);
        _grid->requestSDF();
        _grid->requestGradient();
    }

    value.getField("offset", _offset);
    value.getField("scale", _scale);
}

rapidjson::Value NonstationaryCovariance::toJson(Allocator& allocator) const {
    return JsonObject{ JsonSerializable::toJson(allocator), allocator,
        "type", "nonstationary",
        "cov", *_stationaryCov,
        "grid", *_grid,
        "offset", _offset,
        "scale", _scale
    };
}

void NonstationaryCovariance::loadResources() {
    _grid->loadResources();
    _stationaryCov->loadResources();

    auto gridTr = _grid->invNaturalTransform();
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            _invGridTransformD(r, c) = gridTr(r, c);
        }
    }
}

FloatD NonstationaryCovariance::sampleGrid(Vec3Diff a) const {
    FloatD result;
    Vec3f ap = from_diff(a);
    result[0] = _grid->density(ap);
    result[1] = _grid->gradient(ap).dot({ (float)a.x()[1], (float)a.y()[1] , (float)a.z()[1] });
    result[2] = 0; // linear interp
    return result;
}

static inline Vec3Diff mult(const Mat4f& a, const Vec3Diff& b)
{
    return Vec3Diff(
        a(0, 0) * b.x() + a(0, 1) * b.y() + a(0, 2) * b.z() + a(0, 3),
        a(1, 0) * b.x() + a(1, 1) * b.y() + a(1, 2) * b.z() + a(1, 3),
        a(2, 0) * b.x() + a(2, 1) * b.y() + a(2, 2) * b.z() + a(2, 3)
    );
}

FloatD NonstationaryCovariance::cov(Vec3Diff a, Vec3Diff b) const {

    FloatD sigmaA = (sampleGrid(mult(_grid->invNaturalTransform(), a)) + _offset) * _scale;
    FloatD sigmaB = (sampleGrid(mult(_grid->invNaturalTransform(), b)) + _offset) * _scale;

    return sigmaA * sigmaB * _stationaryCov->cov(a, b);
}

float NonstationaryCovariance::cov(Vec3f a, Vec3f b) const {
    float sigmaA = (_grid->density(_grid->invNaturalTransform() * a) + _offset) * _scale;
    float sigmaB = (_grid->density(_grid->invNaturalTransform() * b) + _offset) * _scale;

    return sigmaA * sigmaB * _stationaryCov->cov(a, b);
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


float TabulatedMean::mean(Vec3f a) const {
    Vec3f p = _grid->invNaturalTransform() * a;
    return (_grid->density(p) + _offset) * _scale;
}

Vec3f TabulatedMean::dmean_da(Vec3f a) const {
    Vec3f p = _grid->invNaturalTransform() * a;
    return _scale* _grid->naturalTransform().transformVector(_grid->gradient(p));
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


float MeshSdfMean::mean(Vec3f a) const {
    // perform a closest point query
    Eigen::MatrixXd V_vis(1, 3);
    V_vis(0, 0) = a.x();
    V_vis(1, 0) = a.y();
    V_vis(2, 0) = a.z();

    Eigen::VectorXd S_vis;
    igl::signed_distance_fast_winding_number(V_vis, V, F, tree, fwn_bvh, S_vis);

    return (float)S_vis(0);
}

Vec3f MeshSdfMean::dmean_da(Vec3f a) const {
    float eps = 0.001f;
    float vals[] = {
        mean(a),
        mean(a + Vec3f(eps, 0.f, 0.f)),
        mean(a + Vec3f(0.f, eps, 0.f)),
        mean(a + Vec3f(0.f, 0.f, eps))
    };

    return Vec3f(vals[1] - vals[0], vals[2] - vals[0], vals[3] - vals[0]) / eps;
}

void MeshSdfMean::loadResources() {

    std::vector<Vertex> _verts;
    std::vector<TriangleI> _tris;

    if (_path && MeshIO::load(*_path, _verts, _tris)) {
#if 0
        _scene = std::make_shared<fcpw::Scene<3>>();

        // set the types of primitives the objects in the scene contain;
        // in this case, we have a single object consisting of only triangles
        _scene->setObjectTypes({ {fcpw::PrimitiveType::Triangle} });

        // set the vertex and triangle count of the (0th) object
        _scene->setObjectVertexCount(_verts.size(), 0);
        _scene->setObjectTriangleCount(_tris.size() , 0);

        _scene->getSceneData()->soups[0].vNormals.resize(_verts.size());

        // specify the vertex positions
        for (int i = 0; i < _verts.size(); i++) {
            Vec3f tpos = _configTransform * _verts[i].pos();
            _scene->setObjectVertex({ tpos.x(), tpos.y(), tpos.z() }, i, 0);
            
            Vec3f tnorm = _configTransform.transformVector(_verts[i].normal());
            _scene->getSceneData()->soups[0].vNormals[i] = { tnorm.x(), tnorm.y(), tnorm.z() };
        }

        // specify the triangle indices
        for (int i = 0; i < _tris.size(); i++) {
            int tri[] = {
                _tris[i].v0,
                _tris[i].v1,
                _tris[i].v2,
            };

            _scene->setObjectTriangle(tri, i, 0);
        }

        //_scene->computeObjectNormals(0);

// now that the geometry has been specified, build the acceleration structure
        _scene->build(fcpw::AggregateType::Bvh_SurfaceArea, true); // the second boolean argument enables vectorization
#endif
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
            _globalCondPs.push_back(v.pos());
            _globalCondDerivs.push_back(Derivative::None);
            _globalCondDerivDirs.push_back(v.normal());
            _globalCondValues.push_back(0);

            _globalCondPs.push_back(v.pos());
            _globalCondDerivs.push_back(Derivative::First);
            _globalCondDerivDirs.push_back(v.normal());
            _globalCondValues.push_back(1);
        }
    }

}

std::tuple<Eigen::VectorXd, CovMatrix> GaussianProcess::mean_and_cov(
    const Vec3f* points, const Derivative* derivative_types, const Vec3f* ddirs,
    Vec3f deriv_dir, size_t numPts) const {

    Eigen::VectorXd ps_mean(numPts);
    CovMatrix ps_cov(numPts, numPts);

#ifdef SPARSE_COV
    std::vector<Eigen::Triplet<float>> tripletList;
    tripletList.reserve(min(numPts * numPts / 10, (size_t)10000));
#endif

    for (size_t i = 0; i < numPts; i++) {
        ps_mean(i) = (*_mean)(derivative_types[i], points[i], deriv_dir);
        const Vec3f& ddir_a = ddirs ? ddirs[i] : deriv_dir;

        for (size_t j = 0; j < numPts; j++) {
            const Vec3f& ddir_b = ddirs ? ddirs[j] : deriv_dir;
            float cov_ij = (*_cov)(derivative_types[i], derivative_types[j], points[i], points[j], ddir_a, ddir_b);

#ifdef SPARSE_COV
            if (std::abs(cov_ij) > _covEps) {
                tripletList.push_back(Eigen::Triplet<float>(i, j, cov_ij));
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
    const Vec3f* points, const Derivative* derivative_types, const Vec3f* ddirs,
    Vec3f deriv_dir, size_t numPts) const {
    Eigen::VectorXd ps_mean(numPts);
    for (size_t i = 0; i < numPts; i++) {
        const Vec3f& ddir= ddirs ? ddirs[i] : deriv_dir;
        ps_mean(i) = (*_mean)(derivative_types[i], points[i], deriv_dir);
    }
    return ps_mean;
}

CovMatrix GaussianProcess::cov(
    const Vec3f* points_a, const Vec3f* points_b, 
    const Derivative* dtypes_a, const Derivative* dtypes_b,
    const Vec3f* ddirs_a, const Vec3f* ddirs_b,
    Vec3f deriv_dir, size_t numPtsA, size_t numPtsB) const {
    CovMatrix ps_cov(numPtsA, numPtsB);

#ifdef SPARSE_COV
    std::vector<Eigen::Triplet<float>> tripletList;
    tripletList.reserve(min(numPtsA * numPtsB / 10, (size_t)10000));
#endif


    for (size_t i = 0; i < numPtsA; i++) {
        const Vec3f& ddir_a = ddirs_a ? ddirs_a[i] : deriv_dir;
        for (size_t j = 0; j < numPtsB; j++) {
            const Vec3f& ddir_b = ddirs_b ? ddirs_b[j] : deriv_dir;

            float cov_ij = (*_cov)(dtypes_a[i], dtypes_b[j], points_a[i], points_b[j], ddir_a, ddir_b);

#ifdef SPARSE_COV
            if (std::abs(cov_ij) > _covEps) {
                tripletList.push_back(Eigen::Triplet<float>(i, j, cov_ij));
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

float GaussianProcess::sample_start_value(Vec3f p, PathSampleGenerator& sampler) const {
    float m = (*_mean)(Derivative::None, p, Vec3f(0.f));
    float sigma = (*_cov)(Derivative::None, Derivative::None, p, p, Vec3f(0.f), Vec3f(0.f));

    return max(0.f, rand_truncated_normal(m, sigma, 0, sampler));
}


Eigen::MatrixXd GaussianProcess::sample(
    const Vec3f* points, const Derivative* derivative_types, size_t numPts,
    const Vec3f* deriv_dirs,
    const Constraint* constraints, size_t numConstraints,
    Vec3f deriv_dir, int samples, PathSampleGenerator& sampler) const {

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
    const Vec3f* points, const Derivative* derivative_types, size_t numPts,
    const Vec3f* deriv_dirs,
    const Vec3f* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
    const Vec3f* cond_deriv_dirs,
    const Constraint* constraints, size_t numConstraints,
    Vec3f deriv_dir, int samples, PathSampleGenerator& sampler) const {

    if (numCondPts == 0) {
        return sample(points, derivative_types, numPts, deriv_dirs, constraints, numConstraints, deriv_dir, samples, sampler);
    }

    std::vector<Vec3f> cond_ps;
    std::vector<Derivative> cond_derivs;
    std::vector<Vec3f> cond_dds;
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

#ifdef SPARSE_COV
    Eigen::SparseQR<CovMatrix, Eigen::AMDOrdering<int>> solver(s11);
    if (solver.info() != Eigen::ComputationInfo::Success) {
        std::cerr << "Conditioning failed!\n";
    }
#else
    Eigen::HouseholderQR<CovMatrix> solver(s11);
#endif

    CovMatrix solved = solver.solve(s12).transpose();


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

// Box muller transform
float GaussianProcess::rand_truncated_normal(float mean, float sigma, float a, PathSampleGenerator& sampler) const {
    if (abs(a - mean) < 0.00001) {
        return abs(mean + sigma * rand_normal_2(sampler).x());
    }

    if (a < mean) {
        while (true) {
            float x = mean + sigma * rand_normal_2(sampler).x();
            if (x >= a) {
                return x;
            }
        }
    }

    float a_bar = (a - mean) / sigma;
    float x_bar;

    while (true) {
        float u = sampler.next1D();
        x_bar = sqrtf(a_bar * a_bar - 2 * log(1 - u));
        float v = sampler.next1D();
        
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