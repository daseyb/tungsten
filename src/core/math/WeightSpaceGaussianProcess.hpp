#ifndef WEIGHTSPACEGAUSSIANPROCESS_HPP_
#define WEIGHTSPACEGAUSSIANPROCESS_HPP_

#include <math/GaussianProcess.hpp>
#include <math/AffineArithmetic.hpp>

namespace Tungsten {

struct WeightSpaceBasis;

struct WeightSpaceRealization {
    std::shared_ptr<WeightSpaceBasis> basis;
    std::shared_ptr<GaussianProcess> gp;
    Eigen::VectorXd weights;

    double evaluate(const Vec3d& p) const;
    Affine<1> evaluate(const Affine<3>& p) const;
    Eigen::VectorXd evaluate(const Vec3d* ps, size_t num_ps) const;
    Vec3d evaluateGradient(const Vec3d& p) const;

    RangeBound rangeBound(const Vec3d& c, const std::vector<Vec3d>& vs) const;

    double lipschitz() const;

    static WeightSpaceRealization sample(std::shared_ptr<WeightSpaceBasis> basis, std::shared_ptr<GaussianProcess> gp, PathSampleGenerator& sampler);
};

struct WeightSpaceBasis {
    Eigen::MatrixXd dirs;
    Eigen::VectorXd freqs;
    Eigen::VectorXd offsets;
    Eigen::VectorXd freqWeights;
    double weightNorm;

    WeightSpaceBasis(int n) {
        dirs.resize(n, 3);
        freqs.resize(n);
        offsets.resize(n);
        freqWeights.resize(n);
        weightNorm = 0.;
    }

    size_t size() const {
        return freqs.rows();
    }

    double evaluate(Eigen::Vector3d p, const Eigen::VectorXd& weights) const;
    Affine<1> evaluate(const Affine<3>& p, const Eigen::VectorXd& weights) const;
    Vec3d evaluateGradient(Eigen::Vector3d p, const Eigen::VectorXd& weights) const;
    Eigen::MatrixXd phi(Eigen::MatrixXd ps, const Eigen::VectorXd& weights) const;

    double lipschitz(const Eigen::VectorXd& weights) const;

    WeightSpaceRealization sampleRealization(std::shared_ptr<GaussianProcess> gp, PathSampleGenerator& sampler) const;

    static WeightSpaceBasis sample(std::shared_ptr<CovarianceFunction> cov, int n, PathSampleGenerator& sampler);
};

}

#endif /* WEIGHTSPACEGAUSSIANPROCESS_HPP_ */