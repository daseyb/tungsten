#include "WeightSpaceGaussianProcess.hpp"

#include <sampling/SampleWarp.hpp>

namespace Tungsten {


Affine::Range sinBound(Affine::Range x) {
    double f_lower = sin(x.lower);
    double f_upper = sin(x.upper);

    // test if there is an interior peak in the range
    x.lower /= 2. * PI;
    x.upper /= 2. * PI;
    bool contains_min = ceil(x.lower - .75) < (x.upper - .75);
    bool contains_max = ceil(x.lower - .25) < (x.upper - .25);

    // result is either at enpoints or maybe an interior peak
    double out_lower = contains_min ? - 1. : min(f_lower, f_upper);
    double out_upper = contains_max ?   1. : max(f_lower, f_upper);

    return { out_lower, out_upper };
}


Affine::Range cosBound(Affine::Range x) {
    return sinBound({ x.lower + PI / 2, x.upper + PI / 2 });
}

Affine sqrt(const Affine& x) {
    return x;
}


// https://github.com/nmwsharp/neural-implicit-queries/blob/main/src/affine_layers.py
Affine aff_sin(const Affine& x) {
    if (x.isConst()) {
        return Affine{ std::sin(x.base), {}};
    }

    auto bx = x.mayContainBounds();

    auto bslope = cosBound(bx);

    double alpha = 0.5 * (bslope.lower + bslope.upper);
    alpha = clamp(alpha, -1., 1.);

    double intA = acos(alpha);
    double intB = -intA;

    auto first = [lower = bx.lower](auto x) { return 2. * PI * ceil((lower + x) / (2. * PI)) - x; };
    auto last = [lower = bx.lower](auto x) { return 2. * PI * ceil((lower + x) / (2. * PI)) - x; };

    double extremes[] = {
        bx.lower, bx.upper, first(intA), last(intA), first(intB), last(intB)
    };

    double r_lower = DBL_MAX;
    double r_upper = -DBL_MAX;

    for (auto& ex : extremes) {
        ex = clamp(ex, bx.lower, bx.upper);
        ex = std::sin(ex) - alpha * ex;

        r_lower = std::min(r_lower, ex);
        r_upper = std::max(r_upper, ex);
    }

    double beta = 0.5 * (r_upper + r_lower);
    double delta = r_upper - beta;

    return x.applyLinearApprox(alpha, beta, delta);
}

Affine aff_cos(const Affine& x) {
    return aff_sin(x + PI / 2);
}

Affine dot(const Eigen::Vector3d& a, const Vec3Aff& b) {
    return b.x() * a.x() + b.y() * a.y() +  b.z() * a.z();
}

Affine spherical_mean(const Vec3Aff& p) {
    Vec3Aff c = { Affine{10.}, Affine{0.}, Affine{0.} };
    Affine r(5);
    auto d = p - c;
    auto l = d.lengthSq();
    return l - r * r;
}

double spherical_mean(const Vec3d& p) {
    Vec3d c = { 10., 0., 0. };
    double r(5);
    auto d = p - c;
    auto l = d.lengthSq();
    return l - r * r;
}

double WeightSpaceRealization::evaluate(const Vec3d& p) const {
    Derivative d = Derivative::None;
    //return sqrt((*gp->_cov)(Derivative::None, Derivative::None, p, p, Vec3d(), Vec3d())) * basis.evaluate(vec_conv<Eigen::Vector3d>(p), weights) + gp->mean(&p, &d, nullptr, Vec3d(0.), 1)(0);
    return spherical_mean(p); // gp->mean(&p, &d, nullptr, Vec3d(0.), 1)(0);
}

Affine WeightSpaceRealization::evaluate(const Vec3Aff& p) const {
    Derivative d = Derivative::None;

    // Assume constant variance
    double scale = sqrt((*gp->_cov)(Derivative::None, Derivative::None, Vec3d(), Vec3d(), Vec3d(), Vec3d()));

    //return basis.evaluate(p, weights) * scale + spherical_mean(p); // +gp->mean(&p, &d, nullptr, Vec3d(0.), 1)(0);
    return spherical_mean(p); // +gp->mean(&p, &d, nullptr, Vec3d(0.), 1)(0);
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

RangeBound WeightSpaceRealization::rangeBound(const Vec3d& c, const Vec3d& v) const {

    Vec3Aff test = {
        Affine{ c.x(), {v.x()} },
        Affine{ c.y(), {v.y()} },
        Affine{ c.z(), {v.z()} },
    };

    Affine res = evaluate(test);

    auto r = res.mayContainBounds();

    if (r.lower > 0) return RangeBound::Positive;
    else if (r.upper < 0) return RangeBound::Negative;
    else return RangeBound::Unknown;
}

Affine WeightSpaceBasis::evaluate(const Vec3Aff& p, const Eigen::VectorXd& weights) const {
    Affine result = 0;
    for (size_t row = 0; row < size(); row++) {
        result = result + aff_cos(dot(dirs.row(row), p) * freqs[row] + offsets[row]) * weights[row];
    }
    return result * sqrt(2. / size());
}

double WeightSpaceBasis::evaluate(Eigen::Vector3d p, const Eigen::VectorXd& weights) const {
    double result = 0;
    for (size_t row = 0; row < size(); row++) {
        result += weights[row] * cos(dirs.row(row).dot(p) * freqs[row] + offsets[row]);
    }
    return result * sqrt(2. / size());
}

Eigen::MatrixXd WeightSpaceBasis::phi(Eigen::MatrixXd ps, const Eigen::VectorXd& weights) const {
    Eigen::MatrixXd phi = Eigen::MatrixXd(ps.rows(), size());
    for (size_t p = 0; p < ps.rows(); p++) {
        for (size_t row = 0; row < size(); row++) {
            phi(p, row) = cos(dirs.row(row).dot(ps.row(p)) * freqs[row] + offsets[row]);
        }
    }
    return phi * sqrt(2. / size());
}

double WeightSpaceBasis::lipschitz(const Eigen::VectorXd& weights) const {
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