#ifndef WEIGHTSPACEGAUSSIANPROCESS_HPP_
#define WEIGHTSPACEGAUSSIANPROCESS_HPP_

#include <math/GaussianProcess.hpp>

namespace Tungsten {

struct WeightSpaceBasis;

struct Affine {

    enum struct Mode {
        Interval,
        AffineFixed,
        AffineTruncate,
        AffineAll,
        AffineAppend
    };

    double base;
    std::vector<double> aff;
    double err = 0;

    Mode mode = Mode::AffineFixed;

    Affine(double base = 0., std::vector<double> aff = {}, double err = 0.) : base(base), aff(aff), err(err) { }


    struct Range {
        double lower, upper;
    };

    bool isConst() const {
        return err == 0 && aff.size() == 0;
    }

    double radius() const {
        if (isConst()) return 0.;
        double res = err;
        for (auto a : aff) {
            res += abs(a);
        }
        return res;
    }

    Range mayContainBounds() const {
        double rad = radius();
        return { base - rad, base + rad };
    }

    //https://github.com/nmwsharp/neural-implicit-queries/blob/main/src/affine.py#L127C11-L127C11
    Affine truncate() const {
        Affine result = *this;

        if (isConst() || mode != Mode::AffineTruncate) {
            return result;
        }

        size_t n_keep = 10;
        if (result.aff.size() < n_keep) {
            return result;
        }

        std::sort(result.aff.begin(), result.aff.end(), [](auto a, auto b) { return abs(a) < abs(b); });

        for (size_t i = n_keep; i < result.aff.size(); i++) {
            result.err += result.aff[i];
        }

        result.aff.resize(n_keep);

        return result;
    }

    Affine applyLinearApprox(double alpha, double beta, double delta) const {
        Affine result = *this;
        result.base = alpha * result.base + beta;

        for (auto& a : result.aff) a *= alpha;

        delta = abs(delta);

        switch (result.mode) {
        case Mode::Interval:
        case Mode::AffineFixed:
            result.err = alpha * result.err + delta;
            return result;
        case Mode::AffineTruncate:
        case Mode::AffineAll:
            result.err = alpha * result.err;
            result.aff.push_back(delta);
            result = result.truncate();
            return result;
        case Mode::AffineAppend:
            result.err = alpha * result.err;
            return result;
        }

    }

    Affine operator-() const
    {
        Affine result;
        result.base = -base;
        result.err = err;
        for (unsigned i = 0; i < aff.size(); ++i)
            result.aff[i] = -aff[i];
        return result;
    }

    static std::pair<Affine, Affine> longer_shorter(const Affine& a, const Affine& b) {
        if (a.aff.size() > b.aff.size()) {
            return { a,b };
        }
        else {
            return { b,a };
        }
    }

    Affine operator+(const Affine& other) const
    {
        auto [longer, shorter] = longer_shorter(*this, other);
        longer.base += shorter.base;
        for (unsigned i = 0; i < shorter.aff.size(); ++i)
            longer.aff[i] += shorter.aff[i];
        longer.err += shorter.err;
        return longer;
    }

    Affine operator-(const Affine& other) const
    {
        auto [longer, shorter] = longer_shorter(*this, other);
        longer.base -= shorter.base;
        for (unsigned i = 0; i < shorter.aff.size(); ++i)
            longer.aff[i] -= shorter.aff[i];
        longer.err += shorter.err;
        return longer;
    }

    Affine operator*(const Affine& other) const
    {
        auto [a, b] = longer_shorter(*this, other);
        if (a.aff.size() == 0) {
            a.aff.resize(1, 0.);
        }

        b.aff.resize(a.aff.size(), 0);

        float s = abs(a.aff[0]);
        float t = abs(b.aff[0]);
        float w = a.aff[0] * b.aff[0];
        float u = s;
        float v = t;
        return Affine(
            a.base * b.base + 0.5 * w,
            { a.base * b.aff[0] + a.aff[0] * b.base },
            a.err * b.err
            + b.err * (abs(a.base) + u)
            + a.err * (abs(b.base) + v)
            + u * v
            - 0.5 * s * t);
    }


    Affine operator+(const double& a) const
    {
        Affine result = *this;
        result.base += a;
        return result;
    }

    Affine operator-(const double& a) const
    {
        Affine result = *this;
        result.base -= a;
        return result;
    }

    Affine operator*(const double& a) const
    {
        Affine result = *this;
        result.base *= a;
        for (unsigned i = 0; i < result.aff.size(); ++i)
            result.aff[i] *= a;
        result.err *= abs(a);
        return result;
    }

    Affine operator/(const double& a) const
    {
        Affine result = *this;
        result.base /= a;
        for (unsigned i = 0; i < result.aff.size(); ++i)
            result.aff[i] /= a;
        result.err /= abs(a);
        return result;
    }

    Affine operator+=(const Affine& other)
    {
        auto [longer, shorter] = longer_shorter(*this, other);
        longer.base += shorter.base;
        for (unsigned i = 0; i < shorter.aff.size(); ++i)
            longer.aff[i] += shorter.aff[i];
        longer.err += shorter.err;
        *this = longer;
        return *this;
    }

    Affine operator-=(const Affine& other)
    {
        auto [longer, shorter] = longer_shorter(*this, other);
        longer.base -= shorter.base;
        for (unsigned i = 0; i < shorter.aff.size(); ++i)
            longer.aff[i] -= shorter.aff[i];
        longer.err += shorter.err;
        *this = longer;
        return *this;
    }

    //Affine operator*=(const Affine& other)
};

Affine sqrt(const Affine& x);

using Vec3Aff = Vec<Affine, 3>;

enum struct RangeBound {
    Unknown,
    Negative,
    Positive
};

struct WeightSpaceRealization {
    const WeightSpaceBasis& basis;
    std::shared_ptr<GaussianProcess> gp;
    Eigen::VectorXd weights;

    double evaluate(const Vec3d& p) const;
    Affine evaluate(const Vec3Aff& p) const;
    Eigen::VectorXd evaluate(const Vec3d* ps, size_t num_ps) const;

    RangeBound rangeBound(const Vec3d& c, const Vec3d& v) const;

    double lipschitz() const;
};

struct WeightSpaceBasis {
    Eigen::MatrixXd dirs;
    Eigen::VectorXd freqs;
    Eigen::VectorXd offsets;

    WeightSpaceBasis(int n) {
        dirs.resize(n, 3);
        freqs.resize(n);
        offsets.resize(n);
    }

    size_t size() const {
        return freqs.rows();
    }

    double evaluate(Eigen::Vector3d p, const Eigen::VectorXd& weights) const;
    Affine evaluate(const Vec3Aff& p, const Eigen::VectorXd& weights) const;
    Eigen::MatrixXd phi(Eigen::MatrixXd ps, const Eigen::VectorXd& weights) const;

    double lipschitz(const Eigen::VectorXd& weights) const;

    WeightSpaceRealization sampleRealization(std::shared_ptr<GaussianProcess> gp, PathSampleGenerator& sampler) const;

    static WeightSpaceBasis sample(std::shared_ptr<CovarianceFunction> cov, int n, PathSampleGenerator& sampler);
};


    
}

#endif /* WEIGHTSPACEGAUSSIANPROCESS_HPP_ */