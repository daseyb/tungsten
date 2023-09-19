#include <core/media/GaussianProcessMedium.hpp>
#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <tinyformat/tinyformat.hpp>
#include <sampling/SampleWarp.hpp>
#include <sampling/SampleWarp.hpp>

#include <core/math/WeightSpaceGaussianProcess.hpp>

using namespace Tungsten;


void compute_realization(std::shared_ptr<GaussianProcess> gp, size_t res, const WeightSpaceRealization& real) {

    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }


    std::vector<Vec3d> points(res * res);
    std::vector<Derivative> derivs(res * res, Derivative::None);

    Eigen::VectorXd samples;
    samples.resize(res * res);

    std::vector<double> grid_xs(res);

    {
        int idx = 0;
        for (int i = 0; i < res; i++) {
            double px = 20. * (double(i) / (res - 1) - 0.5);
            grid_xs[i] = px;

            for (int j = 0; j < res; j++) {
                double py = 20. * (double(j) / (res - 1) - 0.5);
                points[idx] = Vec3d(px, py, 0.);

                samples[idx] = real.evaluate(points[idx]);
                idx++;
            }
        }


        //auto other_samples = real.evaluate(points.data(), points.size());
        //samples -= other_samples;

        //Eigen::VectorXd samples = real.evaluate(points.data(), points.size());

        {
            std::ofstream xfile(
                (basePath / Path("grid-samples.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
            xfile.close();
        }

        {
            std::ofstream xfile(
                (basePath / Path("grid-coordinates.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)grid_xs.data(), sizeof(double) * grid_xs.size());
            xfile.close();
        }
    }

}

double largestSphereAA(std::function<Affine<1>(Affine<3>)> implicit, Vec3d center) {
    const double delta = 0.001;

    double lower = 0.;
    double upper = 100.;

    std::vector<Vec3d> ps;
    std::vector<Vec3d> ds;

    for (int i = 0; i < 1000; i++) {
        if (upper - lower < delta) {
            return lower;
        }

        double midpoint = (upper + lower) * 0.5;
        auto vs = {
            Vec3d(midpoint, 0., 0.),
            Vec3d(0., midpoint, 0.)
        };

        auto val = implicit(Affine<3>(center, vs));

        if (val.rangeBound() != RangeBound::Unknown) {
            lower = midpoint;
        }
        else {
            upper = midpoint;
        }
    }
    std::cerr << "Ran out of iterations in largest sphere AA." << std::endl;
    return lower;
}

double largestSphereAA(std::shared_ptr<GaussianProcess> gp, const WeightSpaceRealization& real, Vec3d center) {
    const double delta = 0.001;

    double lower = 0.;
    double upper = 100.;

    std::vector<Vec3d> ps;
    std::vector<Vec3d> ds;

    for (int i = 0; i < 1000; i++) {
        if (upper - lower < delta) {
            return lower;
        }

        double midpoint = (upper + lower) * 0.5;
        auto vs = {
            Vec3d(midpoint, 0., 0.),
            Vec3d(0., midpoint, 0.)
        };

        if (real.rangeBound(center, vs) != RangeBound::Unknown) {
            lower = midpoint;
        }
        else {
            upper = midpoint;
        }
    }
    std::cerr << "Ran out of iterations in largest sphere AA." << std::endl;
    return lower;
}

void wos(std::shared_ptr<GaussianProcess> gp, const WeightSpaceRealization& real, Vec3d p, PathSampleGenerator& sampler) {
    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }
    std::vector<Vec3d> ps;
    std::vector<double> ds;

    double d = largestSphereAA(gp, real, p);
    int it = 0;

    ps.push_back(p);
    ds.push_back(d);


    while (++it < 10000 && abs(d) > 0.001) {
        auto samp = SampleWarp::uniformCylinder(sampler.next2D());
        samp.z() = 0;
        p += vec_conv<Vec3d>(samp) * d ;
        d = largestSphereAA(gp, real, p);

        ps.push_back(p);
        ds.push_back(d);
    }

    {
        std::ofstream xfile(
            (basePath / Path("wos-centers.bin")).asString(),
            std::ios::out | std::ios::binary);
        xfile.write((char*)ps.data(), sizeof(ps[0]) * ps.size());
        xfile.close();
    }

    {
        std::ofstream xfile(
            (basePath / Path("wos-radii.bin")).asString(),
            std::ios::out | std::ios::binary);
        xfile.write((char*)ds.data(), sizeof(ds[0]) * ds.size());
        xfile.close();
    }

}

bool intersectRayAA(std::shared_ptr<GaussianProcess> gp, const WeightSpaceRealization& real, const Ray& ray, Vec3d& p) {
    const double sig_0 = (ray.farT() - ray.nearT()) * 0.1f;
    const double delta = 0.001;
    const double np = 1.5;
    const double nm = 0.5;
    
    double t = 0;
    double sig = sig_0;

    auto rd = vec_conv<Vec3d>(ray.dir());

    std::vector<Vec3d> ps;
    std::vector<Vec3d> ds;
    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    p = vec_conv<Vec3d>(ray.pos()) + t * rd;
    double f0 = real.evaluate(p);

    int sign0 = f0 < 0 ? -1 : 1;

    for (int i = 0; i < 2048 * 4; i++) {
        auto p_c = p + (t + ray.nearT() + delta) * rd;
        double f_c = real.evaluate(p_c);
        int signc = f_c < 0 ? -1 : 1;


        if (signc != sign0) {
            ps.push_back(p_c);
            ds.push_back(rd * 0.000001);
            {
                std::ofstream xfile(
                    (basePath / Path("affine-interval-centers.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ps.data(), sizeof(ps[0]) * ps.size());
                xfile.close();
            }

            {
                std::ofstream xfile(
                    (basePath / Path("affine-interval-extends.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ds.data(), sizeof(ds[0]) * ds.size());
                xfile.close();
            }

            return true;
        }

        auto c = p + (t + ray.nearT() + sig * 0.5) * rd;
        auto v = sig * 0.5 * rd;

        double nsig;
        if (real.rangeBound(c, { v }) != RangeBound::Unknown) {
            nsig = sig;
            sig = sig * np;

            ps.push_back(c);
            ds.push_back(v);
        }
        else {
            nsig = 0;
            sig = sig * nm;
        }

        t += max(nsig*0.98, delta);

        if (t >= ray.farT()) {
            {
                std::ofstream xfile(
                    (basePath / Path("affine-interval-centers.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ps.data(), sizeof(ps[0]) * ps.size());
                xfile.close();
            }

            {
                std::ofstream xfile(
                    (basePath / Path("affine-interval-extends.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ds.data(), sizeof(ds[0]) * ds.size());
                xfile.close();
            }

            return false;
        }
    }

    std::cerr << "Ran out of iterations in mean intersect IA." << std::endl;
    return false;
}

bool intersectRaySphereTrace(std::shared_ptr<GaussianProcess> gp, const WeightSpaceRealization& real, const Ray& ray, Vec3d& p) {
    double t = ray.nearT() + 0.0001;
    double L = real.lipschitz();

    std::vector<Vec3d> ps;
    std::vector<double> ds;

    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    for (int i = 0; i < 2048 * 4; i++) {
        p = vec_conv<Vec3d>(ray.pos()) + t * vec_conv<Vec3d>(ray.dir());
        double f = real.evaluate(p) / L;

        ps.push_back(p);
        ds.push_back(f);

        if (f < 0.000000001) {

            {
                std::ofstream xfile(
                    (basePath / Path("ray-points.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ps.data(), sizeof(ps[0]) * ps.size());
                xfile.close();
            }

            {
                std::ofstream xfile(
                    (basePath / Path("ray-distances.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ds.data(), sizeof(ds[0]) * ds.size());
                xfile.close();
            }

            return true;
        }

        t += f;

        if (t >= ray.farT()) {
            return false;
        }
    }

    std::cerr << "Ran out of iterations in mean intersect sphere trace." << std::endl;
    return false;
}

void compute_spectral_density(std::shared_ptr<GaussianProcess> gp) {
    gp->loadResources();

    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    {
        std::vector<double> spectralDensity;
        size_t num_samples = 1000;
        double max_w = 10;
        for (size_t i = 0; i < num_samples; i++) {
            spectralDensity.push_back(gp->_cov->spectral_density(double(i) / num_samples * max_w));
        }

        {
            std::ofstream xfile(
                (basePath / Path("spectral_density.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)spectralDensity.data(), sizeof(double) * spectralDensity.size());
            xfile.close();
        }
    }


    {
        UniformPathSampler sampler(0);
        sampler.next2D();

        std::vector<double> spectralDensitySamples;
        size_t num_samples = 100000;
        for (size_t i = 0; i < num_samples; i++) {
            spectralDensitySamples.push_back(gp->_cov->sample_spectral_density(sampler));
        }

        {
            std::ofstream xfile(
                (basePath / Path("spectral_density_samples.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)spectralDensitySamples.data(), sizeof(double) * spectralDensitySamples.size());
            xfile.close();
        }
    }
}


void gen_data(std::shared_ptr<GaussianProcess> gp) {
    std::cout << gp->_cov->id() << "\n";
    
    compute_spectral_density(gp);
    
    UniformPathSampler sampler(0);
    sampler.next2D();

    auto basis = WeightSpaceBasis::sample(gp->_cov, 100, sampler);
    auto real = basis.sampleRealization(gp, sampler);
    
    std::cout << "L = " << real.lipschitz() << "\n";
    compute_realization(gp, 512, real);

    Ray r(Vec3f(-9.f, -9.f, 0.f), Vec3f(1.f, 1.f, 0.f).normalized(), 0.f, 100.f);
    Vec3d ip;
    intersectRaySphereTrace(gp, real, r, ip);
    intersectRayAA(gp, real, r, ip);
    wos(gp, real, Vec3d(1., 0., 0.), sampler);
}


void test_affine() {

    auto gp = std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(0., 0., 0.), 3.f), std::make_shared<RationalQuadraticCovariance>(100.f, 1.f, 0.1f));

    Affine<3> p(Vec3d(0., 0., 0.), { Vec3d(10., 0., 0.), Vec3d(0., 10., 0.)});

    UniformPathSampler sampler(0);
    sampler.next2D();
    WeightSpaceBasis basis = WeightSpaceBasis::sample(gp->_cov, 2, sampler);

    auto real = basis.sampleRealization(gp, sampler);


    Path basePath = Path("testing/weight-space-affine") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    auto implicit = [&real](Affine<3> p) {
        return real.evaluate(p);
    };

    auto result = implicit(p);

    size_t res = 100;
    std::vector<Vec3d> points(res * res);

    Eigen::VectorXd samples;
    samples.resize(res * res);
    std::vector<double> grid_xs(res);

    auto pbs = p.mayContainBounds();
    {
        int idx = 0;
        for (int i = 0; i < res; i++) {
            for (int j = 0; j < res; j++) {
                points[idx] = vec_conv<Vec3d>(
                    lerp((Eigen::Array3d)pbs.lower, 
                         (Eigen::Array3d)pbs.upper,
                         Eigen::Array3d(double(i) / (res - 1), double(j) / (res - 1), 0.)));

                samples[idx] = real.evaluate(points[idx]);
                idx++;
            }
        }

        {
            std::ofstream xfile(
                (basePath / Path("samples.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
            xfile.close();
        }
    }

    {
        std::ofstream xfile(
            (basePath / Path("result.bin")).asString(),
            std::ios::out | std::ios::binary);
        xfile.write((char*)result.base.data(), sizeof(double) * result.base.size());
        xfile.write((char*)result.aff.data(), sizeof(double) * result.aff.size());
        xfile.write((char*)result.err.data(), sizeof(double) * result.err.size());
        xfile.close();
    }

    {
        std::ofstream xfile(
            (basePath / Path("bounds.bin")).asString(),
            std::ios::out | std::ios::binary);
        auto bounds = result.mayContainBounds();
        xfile.write((char*)bounds.lower.data(), sizeof(double) * bounds.lower.size());
        xfile.write((char*)bounds.upper.data(), sizeof(double) * bounds.upper.size());
        xfile.close();
    }

    std::cout << "===========================\n";
    std::cout << largestSphereAA(implicit, Vec3d(0., 0., 0.)) << "\n";

}

int main() {

    test_affine();
    return 0;

    gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(0., 0., 0.), 3.f), std::make_shared<RationalQuadraticCovariance>(1.f, 1.f, 0.1f)));
    gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(0., 0., 0.), 3.f), std::make_shared<RationalQuadraticCovariance>(10.f, 1.f, 0.1f)));
    gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(0., 0., 0.), 3.f), std::make_shared<RationalQuadraticCovariance>(100.f, 1.f, 0.1f)));
    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(10., 0., 0.), 5.f), std::make_shared<RationalQuadraticCovariance>(10.f, 1.f, 0.1f)));
    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(10., 0., 0.), 5.f), std::make_shared<RationalQuadraticCovariance>(1.f, 0.2f, 0.1f)));


    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<SquaredExponentialCovariance>(1.f, 0.5f)));
    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 0.5f, 1.0f)));
    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 0.5f, 0.5f)));
    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 0.5f, 5.0f)));
}
