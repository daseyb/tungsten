#include "GaussianProcess.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "io/MeshIO.hpp"

#include "ziggurat_constants.h"
#include "primitives/Triangle.hpp"
#include "primitives/Vertex.hpp"

#include <fcpw/fcpw.h>

namespace Tungsten {

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
}

rapidjson::Value MeshSdfMean::toJson(Allocator& allocator) const {
    return JsonObject{ JsonSerializable::toJson(allocator), allocator,
        "type", "mesh",
        "file", *_path
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
    if (_path && !MeshIO::load(*_path, _verts, _tris)) {
        _scene = std::make_shared<fcpw::Scene<3>>();

        // set the types of primitives the objects in the scene contain;
        // in this case, we have a single object consisting of only triangles
        _scene->setObjectTypes({ {fcpw::PrimitiveType::Triangle} });

        // set the vertex and triangle count of the (0th) object
        _scene->setObjectVertexCount(_verts.size(), 0);
        _scene->setObjectTriangleCount(_tris.size() , 0);

        // specify the vertex positions
        for (int i = 0; i < _verts.size(); i++) {
            _scene->setObjectVertex({ _verts[i].pos().x(), _verts[i].pos().y(), _verts[i].pos().z() }, i, 0);
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
}

rapidjson::Value GaussianProcess::toJson(Allocator& allocator) const {
    return JsonObject{ JsonSerializable::toJson(allocator), allocator,
        "type", "standard",
        "mean", *_mean,
        "covariance", *_cov
    };
}

void GaussianProcess::loadResources() {
    _mean->loadResources();
    _cov->loadResources();
}

std::tuple<Eigen::VectorXf, Eigen::MatrixXf> GaussianProcess::mean_and_cov(const Vec3f* points, const Derivative* derivative_types, Vec3f deriv_dir, int numPts) const {
    Eigen::VectorXf ps_mean(numPts);
    Eigen::MatrixXf ps_cov(numPts, numPts);

    for (size_t i = 0; i < numPts; i++) {
        ps_mean(i) = (*_mean)(derivative_types[i], points[i], deriv_dir);

        for (size_t j = 0; j < numPts; j++) {
            ps_cov(i, j) = (*_cov)(derivative_types[i], derivative_types[j], points[i], points[j]);
        }
    }

    return { ps_mean, ps_cov };
}

Eigen::VectorXf GaussianProcess::mean(const Vec3f* points, const Derivative* derivative_types, Vec3f deriv_dir, int numPts) const {
    Eigen::VectorXf ps_mean(numPts);
    for (size_t i = 0; i < numPts; i++) {
        ps_mean(i) = (*_mean)(derivative_types[i], points[i], deriv_dir);
    }
    return ps_mean;
}

Eigen::MatrixXf GaussianProcess::cov(const Vec3f* points_a, const Vec3f* points_b, const Derivative* dtypes_a, const Derivative* dtypes_b, int numPtsA, int numPtsB) const {
    Eigen::MatrixXf ps_cov(numPtsA, numPtsB);

    for (size_t i = 0; i < numPtsA; i++) {
        for (size_t j = 0; j < numPtsB; j++) {
            ps_cov(i, j) = (*_cov)(dtypes_a[i], dtypes_b[j], points_a[i], points_b[j]);
        }
    }

    return ps_cov;
}

float GaussianProcess::sample_start_value(Vec3f p, PathSampleGenerator& sampler) const {
    float m = (*_mean)(Derivative::None, p, Vec3f(0.f));
    float sigma = (*_cov)(Derivative::None, Derivative::None, p, p);

    return max(0.f, rand_truncated_normal(m, sigma, 0, sampler));
}


Eigen::MatrixXf GaussianProcess::sample(
    const Vec3f* points, const Derivative* derivative_types, int numPts, 
    const Constraint* constraints, int numConstraints, 
    Vec3f deriv_dir, int samples, PathSampleGenerator& sampler) const {

    auto [ps_mean, ps_cov] = mean_and_cov(points, derivative_types, deriv_dir, numPts);

    Eigen::MatrixXf s = sample_multivariate_normal(ps_mean, ps_cov, constraints, numConstraints, samples, sampler);

    return s;
}

Eigen::MatrixXf GaussianProcess::sample_cond(
    const Vec3f* points, const Derivative* derivative_types, int numPts,
    const Vec3f* cond_points, const float* cond_values, const Derivative* cond_derivative_types, int numCondPts,
    const Constraint* constraints, int numConstraints,
    Vec3f deriv_dir, int samples, PathSampleGenerator& sampler) const {

    Eigen::MatrixXf s11 = cov(cond_points, cond_points, cond_derivative_types, cond_derivative_types, numCondPts, numCondPts);
    Eigen::MatrixXf s12 = cov(cond_points, points, cond_derivative_types, derivative_types, numCondPts, numPts);

    //Eigen::LLT<Eigen::MatrixXf> solver(s11);
    Eigen::MatrixXf solved = s11.colPivHouseholderQr().solve(s12).transpose(); //  solver.solve(s12).transpose();

    Eigen::Map<const Eigen::VectorXf> cond_values_view(cond_values, numCondPts);
    Eigen::VectorXf m2 = mean(points, derivative_types, deriv_dir, numPts) + (solved * (cond_values_view - mean(cond_points, cond_derivative_types, deriv_dir, numCondPts)));

    Eigen::MatrixXf s22 = cov(points, points, derivative_types, derivative_types, numPts, numPts);

    Eigen::MatrixXf s2 = s22 - (solved * s12);

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
    const Eigen::VectorXf& mean, const Eigen::MatrixXf& cov,
    const Constraint* constraints, int numConstraints,
    int samples, PathSampleGenerator& sampler) const {

    // Use the Cholesky decomposition to transform the standard normal vector to the desired multivariate normal distribution
    Eigen::LLT<Eigen::MatrixXf> chol(cov);
    Eigen::MatrixXf normTransform;

    // We can only use the cholesky decomposition if 
      // the covariance matrix is symmetric, pos-definite.
      // But a covariance matrix might be pos-semi-definite.
      // In that case, we'll go to an EigenSolver
    if (chol.info() == Eigen::Success) {
        // Use cholesky solver
        normTransform = chol.matrixL();
    }
    else {
        // Use eigen solver
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigenSolver(cov);
        normTransform = eigenSolver.eigenvectors()
            * eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
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