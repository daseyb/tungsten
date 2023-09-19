#include "WeightSpaceGaussianProcess.hpp"

#include <sampling/SampleWarp.hpp>

namespace Tungsten {

Affine<1> spherical_mean(const Affine<3>& p) {
    Affine<3> c = { Vec3d(0.) };
    Affine<1> r(5.);
    auto d = p - c;
    auto l = d.length();
    return l - r;
}

Affine<1> linear_mean(const Affine<3>& p) {
    Affine<3> c = { Vec3d(0.) };
    auto d = p - c;
    return d.x() - 2;
}

double spherical_mean(const Vec3d& p) {
    Vec3d c = { 0., 0., 0. };
    double r(5);
    auto d = p - c;
    auto l = d.length();
    return l - r;
}

double linear_mean(const Vec3d& p) {
    Vec3d c = { 0., 0., 0. };
    auto d = p - c;
    return d.x()-2;
}

double WeightSpaceRealization::evaluate(const Vec3d& p) const {
    Derivative d = Derivative::None;
    double scale = sqrt((*gp->_cov)(Derivative::None, Derivative::None, Vec3d(), Vec3d(), Vec3d(), Vec3d()));
    //return sqrt((*gp->_cov)(Derivative::None, Derivative::None, p, p, Vec3d(), Vec3d())) * basis.evaluate(vec_conv<Eigen::Vector3d>(p), weights) + gp->mean(&p, &d, nullptr, Vec3d(0.), 1)(0);
    return scale * basis.evaluate(vec_conv<Eigen::Vector3d>(p), weights) + spherical_mean(p);
    //return spherical_mean(p); // gp->mean(&p, &d, nullptr, Vec3d(0.), 1)(0);
}

Affine<1> WeightSpaceRealization::evaluate(const Affine<3>& p) const {
    Derivative d = Derivative::None;
    
    // Assume constant variance
    double scale = sqrt((*gp->_cov)(Derivative::None, Derivative::None, Vec3d(), Vec3d(), Vec3d(), Vec3d()));
    auto basisRes = basis.evaluate(p, weights) * scale; 
    auto np = p;
    np.aff.resize(Eigen::NoChange, std::max(basisRes.aff.cols(), p.aff.cols()));
    np.aff.setZero();
    np.aff.block(0, 0, 3, p.aff.cols()) = p.aff; 

    basisRes += spherical_mean(np); // +gp->mean(&p, &d, nullptr, Vec3d(0.), 1)(0);
    //return spherical_mean(p); // +gp->mean(&p, &d, nullptr, Vec3d(0.), 1)(0);
    return basisRes;
}


Eigen::VectorXd WeightSpaceRealization::evaluate(const Vec3d* ps, size_t num_ps) const {
    Eigen::VectorXd result(num_ps);
    for (size_t p = 0; p < num_ps; p++) {
        result[p] = evaluate(ps[p]);
    }
    return result;
}

double WeightSpaceRealization::lipschitz() const {
    //return sqrt((*gp->_cov)(Derivative::None, Derivative::None, Vec3d(), Vec3d(), Vec3d(), Vec3d())) * basis.lipschitz(weights) + gp->_mean->lipschitz();
    return gp->_mean->lipschitz();
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
        result += aff_cos(dot<3,double>(dirs.row(row), np) * freqs[row] + offsets[row]) * weights[row];
        np.aff.resize(Eigen::NoChange, result.aff.cols());
        np.aff.setZero();
        np.aff.block(0, 0, 3, p.aff.cols()) = p.aff;
    }
    return result * sqrt(2. / size());
}

double WeightSpaceBasis::evaluate(Eigen::Vector3d p, const Eigen::VectorXd& weights) const {
    if (size() == 0) return 0;
    double result = 0;
    for (size_t row = 0; row < size(); row++) {
        result += weights[row] * cos(dirs.row(row).dot(p) * freqs[row] + offsets[row]);
    }
    return result * sqrt(2. / size());
}

Eigen::MatrixXd WeightSpaceBasis::phi(Eigen::MatrixXd ps, const Eigen::VectorXd& weights) const {
    Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(ps.rows(), size());
    if (size() == 0) return phi;
    for (size_t p = 0; p < ps.rows(); p++) {
        for (size_t row = 0; row < size(); row++) {
            phi(p, row) = cos(dirs.row(row).dot(ps.row(p)) * freqs[row] + offsets[row]);
        }
    }
    return phi * sqrt(2. / size());
}

double WeightSpaceBasis::lipschitz(const Eigen::VectorXd& weights) const {
    if (size() == 0) return 0;
    double lipschitz = 0;
    for (size_t row = 0; row < size(); row++) {
        lipschitz += std::abs(weights[row] * freqs[row]);
    }
    return lipschitz * sqrt(2. / size());
}


WeightSpaceBasis WeightSpaceBasis::sample(std::shared_ptr<CovarianceFunction> cov, int n, PathSampleGenerator& sampler) {
    WeightSpaceBasis b(n);
    for (int i = 0; i < n; i++) {
        b.offsets(i) = sampler.next1D() * TWO_PI;
        b.freqs(i) = cov->sample_spectral_density(sampler);
        b.dirs.row(i) = vec_conv<Eigen::Vector3d>(SampleWarp::uniformSphere(sampler.next2D()));
    }
    return b;
}

WeightSpaceRealization WeightSpaceBasis::sampleRealization(std::shared_ptr<GaussianProcess> gp, PathSampleGenerator& sampler) const {
    return WeightSpaceRealization{
        *this, gp, sample_standard_normal(size(), sampler)
    };
}

}