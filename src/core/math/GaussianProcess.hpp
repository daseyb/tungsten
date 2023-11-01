#ifndef GAUSSIANPROCESS_HPP_
#define GAUSSIANPROCESS_HPP_
#include "sampling/PathSampleGenerator.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "BitManip.hpp"
#include "math/Vec.hpp"
#include "math/MathUtil.hpp"
#include "math/Angle.hpp"
#include "sampling/SampleWarp.hpp"
#include <math/GPFunctions.hpp>
#include <sampling/Gaussian.hpp>

#include <functional>
#include <vector>

#include "io/JsonSerializable.hpp"
#include "io/JsonObject.hpp"

namespace Tungsten {

    class GaussianProcess : public JsonSerializable {
    public:

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

        CovMatrix cov_sym(
            const Vec3d* points_a,
            const Derivative* dtypes_a,
            const Vec3d* ddirs_a,
            Vec3d deriv_dir, size_t numPtsA) const;

        double sample_start_value(Vec3d p, PathSampleGenerator& sampler) const;

        MultivariateNormalDistribution create_mvn_cond(
            const Vec3d* points, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Vec3d* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
            const Vec3d* cond_ddirs,
            Vec3d deriv_dir) const;

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


        double eval(
            const Vec3d* points, const double* values, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            Vec3d deriv_dir) const;


        double eval_cond(
            const Vec3d* points, const double* values, const Derivative* derivative_types, size_t numPts,
            const Vec3d* ddirs,
            const Vec3d* cond_points, const double* cond_values, const Derivative* cond_derivative_types, size_t numCondPts,
            const Vec3d* cond_ddirs,
            Vec3d deriv_dir) const;


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

        double noIntersectBound(Vec3d p = Vec3d(0.), double q = 0.9999) const;
        double goodStepsize(Vec3d p = Vec3d(0.), double targetCov = 0.95, Vec3d rd = Vec3d(1., 0., 0.)) const;

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
}

#endif /* GAUSSIANPROCESS_HPP_ */