#include "WeightSpaceGaussianProcess.hpp"

#include <sampling/SampleWarp.hpp>
#include <math/TangentFrame.hpp>

namespace Tungsten {

double WeightSpaceRealization::evaluate(const Vec3d& p) const {
    Derivative d = Derivative::None;
    double c = (*gp->_cov)(Derivative::None, Derivative::None, Vec3d(), Vec3d(), Vec3d(), Vec3d());
    double scale = sqrt(c);
    return scale * basis->evaluate(vec_conv<Eigen::Vector3d>(p), weights) + gp->mean(&p, &d, nullptr, Vec3d(0.), 1)(0);
}

Affine<1> WeightSpaceRealization::evaluate(const Affine<3>& p) const {
    Derivative d = Derivative::None;
    
    // Assume constant variance
    double scale = sqrt((*gp->_cov)(Derivative::None, Derivative::None, Vec3d(), Vec3d(), Vec3d(), Vec3d()));
    auto basisRes = basis->evaluate(p, weights) * scale; 
    auto np = p;
    np.aff.resize(Eigen::NoChange, std::max(basisRes.aff.cols(), p.aff.cols()));
    np.aff.setZero();
    np.aff.block(0, 0, 3, p.aff.cols()) = p.aff; 

    basisRes += gp->_mean->mean(np);
    return basisRes;
}

Vec3d WeightSpaceRealization::evaluateGradient(const Vec3d& p) const {
    Derivative d = Derivative::None;
    double scale = sqrt((*gp->_cov)(Derivative::None, Derivative::None, Vec3d(), Vec3d(), Vec3d(), Vec3d()));
    return scale * basis->evaluateGradient(vec_conv<Eigen::Vector3d>(p), weights) + gp->_mean->dmean_da(p);
}


Eigen::VectorXd WeightSpaceRealization::evaluate(const Vec3d* ps, size_t num_ps) const {
    Eigen::VectorXd result(num_ps);
    for (size_t p = 0; p < num_ps; p++) {
        result[p] = evaluate(ps[p]);
    }
    return result;
}

double WeightSpaceRealization::lipschitz() const {
    return sqrt((*gp->_cov)(Derivative::None, Derivative::None, Vec3d(), Vec3d(), Vec3d(), Vec3d())) * basis->lipschitz(weights) + gp->_mean->lipschitz();
}

RangeBound WeightSpaceRealization::rangeBound(const Vec3d& c, const std::vector<Vec3d>& vs) const {

    Affine<3> test(c, vs);

    Affine<1> res = evaluate(test);

    auto r = res.mayContainBounds();

    if (r.lower(0) > 0) return RangeBound::Positive;
    else if (r.upper(0) < 0) return RangeBound::Negative;
    else return RangeBound::Unknown;
}

Affine<1> WeightSpaceBasis::evaluate(const Affine<3>& p, const Eigen::VectorXd& weights) const {
    if (size() == 0) return 0;
    auto np = p;

    Affine<1> result = 0;
    for (size_t row = 0; row < size(); row++) {
        result += aff_cos(dot<3,double>(dirs.row(row), np) * freqs[row] + offsets[row]) * weights[row] * freqWeights[row];
        if (result.mode != Affine<1>::Mode::AffineFixed) {
            np.aff.resize(Eigen::NoChange, result.aff.cols());
            np.aff.setZero();
            np.aff.block(0, 0, 3, p.aff.cols()) = p.aff;
        }
    }
    return result * sqrt(2. / weightNorm);
}

double WeightSpaceBasis::evaluate(Eigen::Vector3d p, const Eigen::VectorXd& weights) const {
    if (size() == 0) return 0;
    double result = 0;
    for (size_t row = 0; row < size(); row++) {
        result += weights[row] * cos(dirs.row(row).dot(p) * freqs[row] + offsets[row]) * freqWeights[row];
    }
    return result * sqrt(2. / weightNorm) ;
}

Vec3d WeightSpaceBasis::evaluateGradient(Eigen::Vector3d p, const Eigen::VectorXd& weights) const {
    if (size() == 0) return Vec3d(0.);

    Vec3d result = Vec3d(0.);
    for (size_t row = 0; row < size(); row++) {
        result += vec_conv<Vec3d>(dirs.row(row)) * freqWeights[row] * freqs[row] * weights[row] * -sin(dirs.row(row).dot(p) * freqs[row] + offsets[row]);
    }
    return result * sqrt(2. / weightNorm);
}

Eigen::MatrixXd WeightSpaceBasis::phi(Eigen::MatrixXd ps, const Eigen::VectorXd& weights) const {
    Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(ps.rows(), size());
    if (size() == 0) return phi;
    for (size_t p = 0; p < ps.rows(); p++) {
        for (size_t row = 0; row < size(); row++) {
            phi(p, row) = freqWeights[row] * cos(dirs.row(row).dot(ps.row(p)) * freqs[row] + offsets[row]);
        }
    }
    return phi * sqrt(2. / weightNorm);
}

double WeightSpaceBasis::lipschitz(const Eigen::VectorXd& weights) const {
    if (size() == 0) return 0;
    double lipschitz = 0;
    for (size_t row = 0; row < size(); row++) {
        lipschitz += std::abs(freqWeights[row] * weights[row] * freqs[row]);
    }
    return lipschitz * sqrt(2. / weightNorm);
}


WeightSpaceBasis WeightSpaceBasis::sample(std::shared_ptr<CovarianceFunction> cov, int n, PathSampleGenerator& sampler) {
    WeightSpaceBasis b(n);
    b.weightNorm = 0;
    TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame({0., 1., 0.});

    for (int i = 0; i < n; i++) {
        b.offsets(i) = sampler.next1D() * TWO_PI;
        
#if 0
        b.freqs(i) = cov->sample_spectral_density(sampler);
        b.freqWeights(i) = sqrt(b.freqs(i));
        auto dir = SampleWarp::uniformCylinder(sampler.next2D());
        dir.z() = 0;
        b.dirs.row(i) = vec_conv<Eigen::Vector3d>(dir);
#elif 1
        auto dir2d = cov->sample_spectral_density_2d(sampler);
        auto dir = Vec3d(dir2d.x(), dir2d.y(), 0.);

        b.dirs.row(i) = frame.toGlobal(vec_conv<Eigen::Vector3d>(dir.normalized()));
        b.freqs(i) = dir.length();
        b.freqWeights(i) = 1.;

        if (!std::isfinite(b.freqs(i))) {
            std::cerr << "Sampling error!\n";
        }
#elif 0
        auto dir = cov->sample_spectral_density_3d(sampler) * vec_conv<Vec3d>(cov->_aniso);

        b.dirs.row(i) = vec_conv<Eigen::Vector3d>(dir.normalized());
        b.freqs(i) = dir.length();
        b.freqWeights(i) = 1.;

        if (!std::isfinite(b.freqs(i))) {
            std::cerr << "Sampling error!\n";
        }
#else
        b.freqs(i) = cov->sample_spectral_density(sampler);
        b.freqWeights(i) = 1;
        b.dirs.row(i) = Eigen::Vector3d(sampler.nextBoolean(0.5) ? 1. : -1., 0., 0.);
#endif

        b.weightNorm += b.freqWeights(i) * b.freqWeights(i);
    }
    return b;
}

WeightSpaceRealization WeightSpaceRealization::sample(std::shared_ptr<WeightSpaceBasis> basis, std::shared_ptr<GaussianProcess> gp, PathSampleGenerator& sampler) {
    return WeightSpaceRealization{
        basis, gp, sample_standard_normal(basis->size(), sampler)
    };
}


WeightSpaceRealization WeightSpaceBasis::sampleRealization(std::shared_ptr<GaussianProcess> gp, PathSampleGenerator& sampler) const {
    return WeightSpaceRealization{
        std::make_shared<WeightSpaceBasis>(*this), gp, sample_standard_normal(size(), sampler)
    };
}

}