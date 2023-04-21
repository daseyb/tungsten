#include "GaussianProcess.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "ziggurat_constants.h"

namespace Tungsten {


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

Eigen::MatrixXf GaussianProcess::sample(const Vec3f* points, const Derivative* derivative_types, int numPts,
    const std::vector<Vec3f>& cond_points, const std::vector<float>& cond_values, const std::vector<Derivative>& cond_derivative_types, Vec3f deriv_dir, int samples,
    PathSampleGenerator& sampler) {

    Eigen::VectorXf ps_mean(numPts);
    Eigen::MatrixXf ps_cov(numPts, numPts);

    for (size_t i = 0; i < numPts; i++) {
        ps_mean(i) = (*_mean)(derivative_types[i], points[i], deriv_dir);

        for (size_t j = 0; j < numPts; j++) {
            ps_cov(i, j) = (*_cov)(derivative_types[i], derivative_types[j], points[i], points[j]);
        }
    }

    Eigen::MatrixXf s = sample_multivariate_normal(ps_mean, ps_cov, samples, sampler);

    return s;
}


// Get me some bits
uint64_t GaussianProcess::vec2uint(Vec2f v) {
    return ((uint64_t)BitManip::floatBitsToUint(v.x())) | ((uint64_t)BitManip::floatBitsToUint(v.y()) << 32);
}

// From numpy
double GaussianProcess::random_standard_normal(PathSampleGenerator& sampler) {
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
Vec2f GaussianProcess::rand_normal_2(PathSampleGenerator& sampler) {
    float x, y, r;
    do {
        Vec2f sampl = sampler.next2D();
        x = sampl.x() * 2 - 1;
        y = sampl.y() * 2 - 1;
        r = sampl.lengthSq();
    } while (r >= 1.0f || r == 0.0f);

    float d = (float)sqrtf(-2.0f * log(r) / r);
    return { x * d, y * d };
}

Eigen::MatrixXf GaussianProcess::sample_multivariate_normal(const Eigen::VectorXf& mean, const Eigen::MatrixXf& cov, int samples, PathSampleGenerator& sampler) {

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
    Eigen::VectorXf z = Eigen::VectorXf::Zero(mean.size());
    Eigen::MatrixXf sample(mean.size(), samples);

    for (int j = 0; j < samples; j++) {
        for (int i = 0; i < mean.size(); i += 2) {
            Vec2f norm_samp = rand_normal_2(sampler); // { (float)random_standard_normal(sampler), (float)random_standard_normal(sampler) };
            z(i) = norm_samp.x();
            z(i + 1) = norm_samp.y();
        }

        sample.col(j) = mean + normTransform * z;
    }

    return sample;
}
}