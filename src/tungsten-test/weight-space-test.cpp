#include <core/media/GaussianProcessMedium.hpp>
#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <tinyformat/tinyformat.hpp>

#include <core/math/WeightSpaceGaussianProcess.hpp>

using namespace Tungsten;


void compute_realization(std::shared_ptr<GaussianProcess> gp, size_t res, const WeightSpaceRealization& real) {

    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }


    std::vector<Vec3d> points(res * res);
    std::vector<Derivative> derivs(res * res, Derivative::None);

    {
        int idx = 0;
        for (int i = 0; i < res; i++) {
            for (int j = 0; j < res; j++) {
                points[idx] = 20. * (Vec3d((float)i, (float)j, 0.) / (res - 1) - 0.5);
                points[idx][2] = 0.f;
                idx++;
            }
        }

        Eigen::VectorXd samples = real.evaluate(points.data(), points.size()) + gp->mean(points.data(), derivs.data(), nullptr, Vec3d(0.), points.size());

        {
            std::ofstream xfile(
                (basePath / Path("grid-samples.bin")).asString(),
                std::ios::out | std::ios::binary);
            xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
            xfile.close();
        }
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
    std::vector<double> ds;
    Path basePath = Path("testing/weight-space") / gp->_cov->id();
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    p = vec_conv<Vec3d>(ray.pos()) + t * rd;
    double f0 = real.evaluate(p);

    for (int i = 0; i < 2048 * 4; i++) {
        auto p_c = p + (t + ray.nearT() + delta) * rd;
        double f_c = real.evaluate(p_c);


        if (signbit(f_c) != signbit(f0)) {

            {
                std::ofstream xfile(
                    (basePath / Path("affine-interval-centers.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ps.data(), sizeof(ps[0]) * ps.size());
                xfile.close();
            }

            {
                std::ofstream xfile(
                    (basePath / Path("affine-interval-sizes.bin")).asString(),
                    std::ios::out | std::ios::binary);
                xfile.write((char*)ds.data(), sizeof(ds[0]) * ds.size());
                xfile.close();
            }

            return true;
        }

        auto c = p + (t + ray.nearT() + sig * 0.5) * rd;
        auto v = sig * 0.5 * rd;

        ps.push_back(c);
        ds.push_back(sig);

        double nsig;
        if (real.rangeBound(c, v) != RangeBound::Unknown) {
            nsig = sig;
            sig = sig * np;
        }
        else {
            nsig = 0;
            sig = sig * nm;
        }

        t += max(nsig, delta);

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
                    (basePath / Path("affine-interval-sizes.bin")).asString(),
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

    auto basis = WeightSpaceBasis::sample(gp->_cov, 5000, sampler);
    auto real = basis.sampleRealization(gp, sampler);
    
    std::cout << "L = " << real.lipschitz() << "\n";
    compute_realization(gp, 512, real);

    Ray r(Vec3f(-9.f, 0.f, 0.f), Vec3f(1.f, 0.f, 0.f), 0.f, 100.f);
    Vec3d ip;
    intersectRaySphereTrace(gp, real, r, ip);
    intersectRayAA(gp, real, r, ip);
}

int main() {

    gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(10., 0., 0.), 5.f), std::make_shared<RationalQuadraticCovariance>(1.f, 1.f, 0.1f)));
    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(10., 0., 0.), 5.f), std::make_shared<RationalQuadraticCovariance>(10.f, 1.f, 0.1f)));
    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(Vec3d(10., 0., 0.), 5.f), std::make_shared<RationalQuadraticCovariance>(1.f, 0.2f, 0.1f)));


    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<SquaredExponentialCovariance>(1.f, 0.5f)));
    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 0.5f, 1.0f)));
    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 0.5f, 0.5f)));
    //gen_data(std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<RationalQuadraticCovariance>(1.f, 0.5f, 5.0f)));
}
