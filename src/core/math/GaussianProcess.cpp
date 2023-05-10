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

#ifdef SPARSE_COV
//#define SPARSE_SOLVE
#endif

namespace Tungsten {

FloatD CovarianceFunction::dcov_da(Vec3Diff a, Vec3Diff b, Eigen::Array3d dir) const {
    Eigen::Array3d zd = Eigen::Array3d::Zero();
    auto covDiff = autodiff::derivatives([&](auto a, auto b) { return cov(a,b); }, autodiff::along(dir, zd), at(a, b));
    return covDiff[1];
}

FloatD CovarianceFunction::dcov_db(Vec3Diff a, Vec3Diff b, Eigen::Array3d dir) const {
    return dcov_da(b, a, dir);
}

FloatD CovarianceFunction::dcov2_dadb(Vec3Diff a, Vec3Diff b, Eigen::Array3d dir) const {
    auto covDiff = autodiff::derivatives([&](auto a, auto b) { return cov(a, b); }, autodiff::along(Eigen::Array3d(-dir*0.5f), Eigen::Array3d(dir*0.5f)), at(a, b));
    return -covDiff[2];
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

void TabulatedMean::fromJson(JsonPtr value, const Scene& scene) {
    MeanFunction::fromJson(value, scene);

    if (auto grid = value["grid"]) {
        _grid = scene.fetchGrid(grid);
        _grid->requestSDF();
        _grid->requestGradient();
    }
}

rapidjson::Value TabulatedMean::toJson(Allocator& allocator) const {
    return JsonObject{ JsonSerializable::toJson(allocator), allocator,
        "type", "tabulated",
        "grid", *_grid
    };
}


float TabulatedMean::mean(Vec3f a) const {
    Vec3f p = _grid->invNaturalTransform() * a;
    return _grid->density(p);
}

Vec3f TabulatedMean::dmean_da(Vec3f a) const {
    Vec3f p = _grid->invNaturalTransform() * a;
    return _grid->naturalTransform().transformVector(_grid->gradient(p));
}

void TabulatedMean::loadResources() {
    _grid->loadResources();
}


void MeshSdfMean::fromJson(JsonPtr value, const Scene& scene) {
    MeanFunction::fromJson(value, scene);
    if (auto path = value["file"]) _path = scene.fetchResource(path);
    value.getField("transform", _configTransform);
    _invConfigTransform = _configTransform.invert();

}

rapidjson::Value MeshSdfMean::toJson(Allocator& allocator) const {
    return JsonObject{ JsonSerializable::toJson(allocator), allocator,
        "type", "mesh",
        "file", *_path,
        "transform", _configTransform
    };
}


float MeshSdfMean::mean(Vec3f a) const {
    // perform a closest point query
    fcpw::Interaction<3> interaction;
    if (_scene->findClosestPoint({ a.x(), a.y(), a.z() }, interaction)) {
        return interaction.signedDistance({ a.x(), a.y(), a.z() });
    }
    else {
        return 1000000.f;
    }
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
    if (_path && MeshIO::load(*_path, _verts, _tris)) {
        _scene = std::make_shared<fcpw::Scene<3>>();

        // set the types of primitives the objects in the scene contain;
        // in this case, we have a single object consisting of only triangles
        _scene->setObjectTypes({ {fcpw::PrimitiveType::Triangle} });

        // set the vertex and triangle count of the (0th) object
        _scene->setObjectVertexCount(_verts.size(), 0);
        _scene->setObjectTriangleCount(_tris.size() , 0);

        // specify the vertex positions
        for (int i = 0; i < _verts.size(); i++) {
            Vec3f tpos = _configTransform * _verts[i].pos();
            _scene->setObjectVertex({ tpos.x(), tpos.y(), tpos.z() }, i, 0);
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

        _scene->computeObjectNormals(0);

        // now that the geometry has been specified, build the acceleration structure
        _scene->build(fcpw::AggregateType::Bvh_SurfaceArea, true); // the second boolean argument enables vectorization
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
}

rapidjson::Value GaussianProcess::toJson(Allocator& allocator) const {
    return JsonObject{ JsonSerializable::toJson(allocator), allocator,
        "type", "standard",
        "mean", *_mean,
        "covariance", *_cov,
        "max_num_eigenvalues", _maxEigenvaluesN,
        "covariance_epsilon", _covEps
    };
}

void GaussianProcess::loadResources() {
    _mean->loadResources();
    _cov->loadResources();
}

std::tuple<Eigen::VectorXf, CovMatrix> GaussianProcess::mean_and_cov(const Vec3f* points, const Derivative* derivative_types, Vec3f deriv_dir, size_t numPts) const {
    Eigen::VectorXf ps_mean(numPts);
    CovMatrix ps_cov(numPts, numPts);

#ifdef SPARSE_COV
    std::vector<Eigen::Triplet<float>> tripletList;
    tripletList.reserve(min(numPts * numPts / 10, (size_t)10000));
#endif

    for (size_t i = 0; i < numPts; i++) {
        ps_mean(i) = (*_mean)(derivative_types[i], points[i], deriv_dir);

        for (size_t j = 0; j < numPts; j++) {
            float cov_ij = (*_cov)(derivative_types[i], derivative_types[j], points[i], points[j], deriv_dir);

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
#endif

    return { ps_mean, ps_cov };
}

Eigen::VectorXf GaussianProcess::mean(const Vec3f* points, const Derivative* derivative_types, Vec3f deriv_dir, size_t numPts) const {
    Eigen::VectorXf ps_mean(numPts);
    for (size_t i = 0; i < numPts; i++) {
        ps_mean(i) = (*_mean)(derivative_types[i], points[i], deriv_dir);
    }
    return ps_mean;
}

CovMatrix GaussianProcess::cov(const Vec3f* points_a, const Vec3f* points_b, const Derivative* dtypes_a, const Derivative* dtypes_b, Vec3f deriv_dir, size_t numPtsA, size_t numPtsB) const {
    CovMatrix ps_cov(numPtsA, numPtsB);

#ifdef SPARSE_COV
    std::vector<Eigen::Triplet<float>> tripletList;
    tripletList.reserve(min(numPtsA * numPtsB / 10, (size_t)10000));
#endif


    for (size_t i = 0; i < numPtsA; i++) {
        for (size_t j = 0; j < numPtsB; j++) {
            float cov_ij = (*_cov)(dtypes_a[i], dtypes_b[j], points_a[i], points_b[j], deriv_dir);

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
#endif

    return ps_cov;
}

float GaussianProcess::sample_start_value(Vec3f p, PathSampleGenerator& sampler) const {
    float m = (*_mean)(Derivative::None, p, Vec3f(0.f));
    float sigma = (*_cov)(Derivative::None, Derivative::None, p, p, Vec3f(0.f));

    return max(0.f, rand_truncated_normal(m, sigma, 0, sampler));
}


Eigen::MatrixXf GaussianProcess::sample(
    const Vec3f* points, const Derivative* derivative_types, size_t numPts,
    const Constraint* constraints, size_t numConstraints,
    Vec3f deriv_dir, int samples, PathSampleGenerator& sampler) const {

    auto [ps_mean, ps_cov] = mean_and_cov(points, derivative_types, deriv_dir, numPts);

    Eigen::MatrixXf s = sample_multivariate_normal(ps_mean, ps_cov, constraints, numConstraints, samples, sampler);

    return s;
}

Eigen::MatrixXf GaussianProcess::sample_cond(
    const Vec3f* points, const Derivative* derivative_types, size_t numPts,
    const Vec3f* cond_points, const float* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
    const Constraint* constraints, size_t numConstraints,
    Vec3f deriv_dir, int samples, PathSampleGenerator& sampler) const {

    CovMatrix s11 = cov(cond_points, cond_points, cond_derivative_types, cond_derivative_types, deriv_dir, numCondPts, numCondPts);
    CovMatrix s12 = cov(cond_points, points, cond_derivative_types, derivative_types, deriv_dir, numCondPts, numPts);

#ifdef SPARSE_COV
    Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::AMDOrdering<int>> solver(s11);
    CovMatrix solved = solver.solve(s12).transpose();
#else
    CovMatrix solved = s11.colPivHouseholderQr().solve(s12).transpose();
#endif


    Eigen::Map<const Eigen::VectorXf> cond_values_view(cond_values, numCondPts);
    Eigen::VectorXf m2 = mean(points, derivative_types, deriv_dir, numPts) + (solved * (cond_values_view - mean(cond_points, cond_derivative_types, deriv_dir, numCondPts)));

    CovMatrix s22 = cov(points, points, derivative_types, derivative_types, deriv_dir, numPts, numPts);

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
Vec2f GaussianProcess::rand_normal_2(PathSampleGenerator& sampler) const {
    float u1 = sampler.next1D();
    float u2 = sampler.next1D();

    float r = sqrtf(-2 * log(u1));
    float x = cos(2 * PI * u2);
    float y = sin(2 * PI * u2);
    float z1 = r * x;
    float z2 = r * y;

    return Vec2f(z1, z2);

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

//float GaussianProcess::rand_truncated_normal(float mean, float sigma, float a, PathSampleGenerator& sampler) const {
//    if (abs(a - mean) < 0.00001) {
//        return abs(mean + sigma * rand_normal_2(sampler).x());
//    }
//
//    float a_bar = (a - mean) / sigma;
//    if (!std::isfinite(a_bar)) {
//        std::cout << a << " " << mean << " " << sigma << "\n";
//    }
//    float y_bar;
//
//    while (true) {
//        float u = sampler.next1D();
//        y_bar = -1 / a_bar * log(1 - u);
//        float v = sampler.next1D();
//
//        if (v < expf(-y_bar*y_bar/2)) {
//            break;
//        }
//    }
//
//    return sigma * (y_bar + a_bar) + mean;
//}

Eigen::MatrixXf GaussianProcess::sample_multivariate_normal(
    const Eigen::VectorXf& mean, const CovMatrix& cov,
    const Constraint* constraints, int numConstraints,
    int samples, PathSampleGenerator& sampler) const {

    Eigen::MatrixXf normTransform;

#ifndef SPARSE_SOLVE
    // Use the Cholesky decomposition to transform the standard normal vector to the desired multivariate normal distribution

#ifdef SPARSE_COV
    Eigen::SimplicialLLT<CovMatrix> chol(cov);
#else
    Eigen::LLT<Eigen::MatrixXf> chol(cov);
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
        Spectra::SparseGenMatProd<float> op(cov);
        Spectra::SymEigsSolver<Spectra::SparseGenMatProd<float>> eigs(
            op, 
            min(size_t(cov.rows()-2), _maxEigenvaluesN), 
            min(size_t(cov.rows()-1), _maxEigenvaluesN * 2)
        );

        // Initialize and compute
        eigs.init();
        int nconv = eigs.compute(Spectra::SortRule::LargestMagn);
#else
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigs(cov);
#endif
        normTransform = eigs.eigenvectors() 
            * eigs.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    }

    // Generate a vector of standard normal variates with the same dimension as the mean
    Eigen::VectorXf z = Eigen::VectorXf(mean.size());
    Eigen::MatrixXf sample(mean.size(), samples);

    int numTries = 0;
    for (int j = 0; j < samples; /*only advance sample idx if the sample passes all constraints*/) {

        numTries++;

        // We're always getting two samples, so make use of that
        for (int i = 0; i < mean.size() / 2; i++) {
            Vec2f norm_samp = rand_normal_2(sampler); // { (float)random_standard_normal(sampler), (float)random_standard_normal(sampler) };
            z(i*2) = norm_samp.x();
            z(i*2 + 1) = norm_samp.y();
        }

        // Fill up the last one for an uneven number of samples
        if (mean.size() % 2) {
            Vec2f norm_samp = rand_normal_2(sampler);
            z(mean.size()-1) = norm_samp.x();
        }

        Eigen::VectorXf currSample = mean + normTransform * z;

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