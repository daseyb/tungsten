#include <core/math/GaussianProcess.hpp>
#include <core/media/GaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/Scene.hpp>
#include <thread/ThreadUtils.hpp>
#include <math/WeightSpaceGaussianProcess.hpp>
#include <ccomplex>
#include <fftw3.h>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif

#ifdef CERES_AVAILABLE
#include "ceres/ceres.h"
#endif

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 128;

int gen3d(int argc, char** argv) {

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

    std::string prefix = "csg-two-spheres-nofilter";

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    auto scenePath = Path(argv[1]);
    std::cout << scenePath.parent().asString() << "\n";

    Scene* scene = nullptr;
    TraceableScene* tscene = nullptr;
    try {
        scene = Scene::load(scenePath);
        scene->loadResources();
        tscene = scene->makeTraceable();
    }
    catch (std::exception& e) {
        std::cout << e.what();
        return -1;
    }

    std::shared_ptr<GaussianProcessMedium> gp_medium = std::static_pointer_cast<GaussianProcessMedium>(scene->media()[0]);

    auto gp = std::static_pointer_cast<GPSampleNode>(gp_medium->_gp);

    UniformPathSampler sampler(0);
    sampler.next2D();

    std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> fderivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    auto processBox = scene->findPrimitive("processBox");

    Vec3d min = vec_conv<Vec3d>(processBox->bounds().min());
    Vec3d max = vec_conv<Vec3d>(processBox->bounds().max());

    Eigen::VectorXf mean(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    Eigen::VectorXf variance(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    Eigen::VectorXf aniso(6 * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    
    auto cellSize = (max - min) / NUM_SAMPLE_POINTS;
    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>(cellSize));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(min));

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(100.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("mean");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));

    openvdb::FloatGrid::Accessor meanAccessor = meanGrid->getAccessor();

    int numEstSamples = 100;
    {
        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            std::cout << i << "\r";
#pragma omp parallel for
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    int idx = i * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS + j * NUM_SAMPLE_POINTS + k;
                    points[idx] = lerp(min, max, Vec3d((float)i, (float)j, (float)k) / (NUM_SAMPLE_POINTS - 1));
                    derivs[idx] = Derivative::None;
                    fderivs[idx] = Derivative::First;

                    mean[idx] = 0;
                    std::vector<float> samples(numEstSamples);
                    for (int s = 0; s < numEstSamples; s++) {
                        Vec2d s1 = rand_normal_2(sampler);
                        Vec2d s2 = rand_normal_2(sampler);
                        Vec3d offset = { (float)s1.x(), (float)s1.y(), (float)s2.x() };
                        Vec3d p = points[idx] + offset * cellSize;
                        samples[s] = gp->mean(&p, &derivs[idx], nullptr, Vec3d(1.0f, 0.0f, 0.0f), 1)(0);
                        mean[idx] += samples[s];
                    }

                    mean[idx] /= numEstSamples;
                }
            }
        }

        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            std::cout << i << "\r";
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    int idx = i * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS + j * NUM_SAMPLE_POINTS + k;
                    meanAccessor.setValue({ i,j,k }, mean[idx]);
                }
            }
        }


        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::QuadraticSampler> meanGridSampler(
            meanGrid->tree(), 
            meanGrid->transform());


        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            std::cout << i << "\r";
#pragma omp parallel for
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    int idx = i * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS + j * NUM_SAMPLE_POINTS + k;
                    Vec3d cp = Vec3d((float)i, (float)j, (float)k);

                    std::vector<float> samples(numEstSamples);
                    int valid_samples = 0;
                    for (int s = 0; s < numEstSamples; s++) {
                        Vec2d s1 = rand_normal_2(sampler);
                        Vec2d s2 = rand_normal_2(sampler);
                        Vec3d offset = { (float)s1.x(), (float)s1.y(), (float)s2.x() };
                        Vec3d p = cp + offset;

                        if (p.x() < 0 || p.y() < 0 || p.z() < 0 ||
                            p.x() > NUM_SAMPLE_POINTS-1 || p.y() > NUM_SAMPLE_POINTS - 1 || p.z() > NUM_SAMPLE_POINTS - 1) {
                            samples[s] = 0;
                        }
                        else {
                            float meanSample = meanGridSampler.isSample(openvdb::Vec3R(p.x(), p.y(), p.z()));
                            Vec3d bp = lerp(min, max, p / (NUM_SAMPLE_POINTS - 1));
                            samples[s] = gp->mean(&bp, &derivs[idx], nullptr, Vec3d(1.0f, 0.0f, 0.0f), 1)(0) - meanSample;
                            valid_samples++;
                        }
                    }

                    variance[idx] = 0;
                    if (valid_samples > 1) {
                        for (int s = 0; s < numEstSamples; s++) {
                            variance[idx] += sqr(samples[s]);
                        }
                        variance[idx] /= (valid_samples - 1);
                    }

                    {
                        Vec3d bp = lerp(min, max, cp / (NUM_SAMPLE_POINTS - 1));

                        Vec3d ps[] = {
                            bp,bp,bp
                        };

                        Derivative derivs[]{
                            Derivative::First, Derivative::First, Derivative::First
                        };

                        Vec3d ddirs[] = {
                            Vec3d(1.f, 0.f, 0.f),
                            Vec3d(0.f, 1.f, 0.f),
                            Vec3d(0.f, 0.f, 1.f),
                        };

                        auto gps = gp->mean(ps, derivs, ddirs, Vec3d(0.f), 3);
                        auto grad = vec_conv<Vec3f>(gps);
                        TangentFrame tf(grad);

                        Eigen::Matrix3f vmat;
                        vmat.col(0) = vec_conv<Eigen::Vector3f>(tf.tangent);
                        vmat.col(1) = vec_conv<Eigen::Vector3f>(tf.bitangent);
                        vmat.col(2) = vec_conv<Eigen::Vector3f>(tf.normal);


                        Eigen::Matrix3f smat = Eigen::Matrix3f::Identity();
                        smat.diagonal() = Eigen::Vector3f{ 1.f, 1.f, 5.f };

                        Eigen::Matrix3f mat = vmat * smat * vmat.transpose();

                        aniso[idx * 6 + 0] = mat(0, 0);
                        aniso[idx * 6 + 1] = mat(1, 1);
                        aniso[idx * 6 + 2] = mat(2, 2);

                        aniso[idx * 6 + 3] = mat(0, 1);
                        aniso[idx * 6 + 4] = mat(0, 2);
                        aniso[idx * 6 + 5] = mat(1, 2);
                    }
                }
            }
        }
    }

    {
        auto varGrid = openvdb::createGrid<openvdb::FloatGrid>();
        openvdb::FloatGrid::Accessor varAccessor = varGrid->getAccessor();
        varGrid->setName("density");
        varGrid->setTransform(openvdb::math::Transform::createLinearTransform(4.0 / NUM_SAMPLE_POINTS));
        
        openvdb::GridPtrVec anisoGrids;
        std::vector<openvdb::FloatGrid::Accessor> anisoAccessors;

        std::string names[] = {
            "sigma_xx", "sigma_yy", "sigma_zz",
            "sigma_xy", "sigma_xz", "sigma_yz",
        };

        for (int i = 0; i < 6; i++) {
            auto anisoGrid = openvdb::createGrid<openvdb::FloatGrid>();
            anisoGrid->setName(names[i]);
            anisoGrid->setTransform(openvdb::math::Transform::createLinearTransform(4.0 / NUM_SAMPLE_POINTS));
            anisoAccessors.push_back(anisoGrid->getAccessor());
            anisoGrids.push_back(anisoGrid);
        }


        int idx = 0;
        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            std::cout << i << "\r";
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    varAccessor.setValue({ i,j,k }, variance(idx));

                    for (int anisoIdx = 0; anisoIdx < 6; anisoIdx++) {
                        anisoAccessors[anisoIdx].setValue({ i,j,k }, aniso[idx * 6 + anisoIdx]);
                    }

                    idx++;
                }
            }
        }

        {
            openvdb::GridPtrVec grids;
            grids.push_back(meanGrid);
            grids.push_back(varGrid);
            grids.insert(grids.end(), anisoGrids.begin(), anisoGrids.end());
            openvdb::io::File file(tinyformat::format("./data/testing/load-gen/%s-isotopric-%d.vdb", prefix, NUM_SAMPLE_POINTS));
            file.write(grids);
            file.close();
        }
    }

    return 0;

}

int mesh_convert(int argc, char** argv) {

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    MeshSdfMean mean(std::make_shared<Path>(argv[1]), true);
    mean.loadResources();

    int dim = std::stod(argv[2]);

    auto scenePath = Path(argv[1]);
    std::cout << scenePath.parent().asString() << "\n";

    auto bounds = mean.bounds();
    bounds.grow(bounds.diagonal().length() * 0.2);

    Vec3d minp = bounds.min();
    Vec3d maxp = bounds.max();

    Vec3d extends = (maxp - minp);

    double max_extend = extends.max();

    std::cout << extends << ":" << max_extend << "\n";

    Vec3d aspect = extends / max_extend;

    Vec3i dims = vec_conv<Vec3i>(aspect * dim);

    std::cout << aspect << ":" << dims << "\n";

    auto cellSize = max_extend / dim;

    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>(Vec3d(cellSize)));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(minp));

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(100.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("mean");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));
    openvdb::FloatGrid::Accessor meanAccessor = meanGrid->getAccessor();

    UniformPathSampler sampler(0);
    sampler.next2D();

    int num_samples = 1;

    for (int i = 0; i < dims.x(); i++) {
        std::cout << i << "\r";
        for (int j = 0; j < dims.y(); j++) {
#pragma omp parallel for
            for (int k = 0; k < dims.z(); k++) {
                auto p = lerp(minp, maxp, Vec3d((float)i, (float)j, (float)k) / vec_conv<Vec3d>(dims));

                double m = 0;
                
                if (num_samples > 0) {
                    for (int s = 0; s < num_samples; s++) {
                        Vec2d s1 = rand_normal_2(sampler);
                        Vec2d s2 = rand_normal_2(sampler);
                        Vec3d offset = { (float)s1.x(), (float)s1.y(), (float)s2.x() };
                        Vec3d pt = p + offset * cellSize * 0.1;
                        m += mean(Derivative::None, pt, Vec3d());
                    }
                    m /= num_samples;
                }
                else {
                    m = mean(Derivative::None, p, Vec3d());
                }

#pragma omp critical
                {
                    meanAccessor.setValue({ i,j,k }, m);
                }
            }
        }
    }

    {
        openvdb::GridPtrVec grids;
        grids.push_back(meanGrid);
        openvdb::io::File file(scenePath.stripExtension().asString() + "-" + std::to_string(dim) + ".vdb");
        file.write(grids);
        file.close();
    }
}

std::vector<double> compute_acf_direct(const double* signal, size_t n, double mean) {

    std::vector<double> acf(n / 2);
    
    double var = 0;
    for (size_t i = 0; i < n; i++)
    {
        var += sqr(mean - signal[i]);
    }
    var /= n;

    for (size_t t = 0; t < acf.size(); t++)
    {
        double nu = 0; // Numerator
        double de = 0; // Denominator
        for (size_t i = 0; i < n; i++)
        {
            double xim = signal[i] - mean;
            nu += xim * (signal[(i + t) % n] - mean);
            de += xim * xim;
        }

        acf[t] = var * nu / de;
    }

    return acf;
}

std::vector<double> compute_acf_fftw(const double* signal, size_t n, double dt, double mean) {
    // Allocate memory for the FFTW input and output arrays
    double* in = (double*)fftw_malloc(sizeof(double) * n);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (n / 2 + 1));

    // Create a plan for the forward FFT
    fftw_plan forward_plan = fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);

    // Initialize the input array with the signal
    for (int i = 0; i < n; ++i) {
        in[i] = signal[i] - mean;
    }

    // Execute the forward FFT
    fftw_execute(forward_plan);

    // Compute the power spectrum by squaring the magnitude of the FFT
    for (int i = 0; i < n / 2 + 1; ++i) {
        double magnitude = out[i][0] * out[i][0] + out[i][1] * out[i][1];
        out[i][0] = magnitude / n;
        out[i][1] = 0.0;
    }

    // Create a plan for the backward FFT
    fftw_plan backward_plan = fftw_plan_dft_c2r_1d(n, out, in, FFTW_ESTIMATE);

    // Execute the backward FFT
    fftw_execute(backward_plan);

    // Normalize the result by the size of the signal
    for (int i = 0; i < n; ++i) {
        in[i] /= n;
    }


    // Clean up
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);

    auto result = std::vector(in, in + n);

    fftw_free(in);
    fftw_free(out);

    return result;
}

struct CovResidual {
    CovResidual(double x, double y, const CovarianceFunction* cov) : x_(x), y_(y), cov_(cov) {}

    bool operator()(const double* const sigma, const double* const ls, double* residual) const {
        residual[0] = y_ - sigma[0] * sigma[0] *(*cov_)(
            Derivative::None, Derivative::None, 
            Vec3d(0.), Vec3d(x_, 0., 0.) / ls[0],
            Vec3d(), Vec3d());
        return true;
    }
private:
    const CovarianceFunction* cov_;
    const double x_;
    const double y_;
};

std::tuple<double, double> fit_cov(const double* t, const double* acf, int n, const CovarianceFunction* cov) {
    const double initial_sigma = 1.0;
    const double initial_ls = 1.0;
    double sigma = initial_sigma;
    double ls = initial_ls;
    ceres::Problem problem;
    for (int i = 0; i < n; ++i) {
        problem.AddResidualBlock(
            new ceres::NumericDiffCostFunction<CovResidual, ceres::CENTRAL, 1, 1, 1>(
                new CovResidual(t[i], acf[i], cov)),
            nullptr,
            &sigma,
            &ls);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Initial m: " << initial_sigma << " c: " << initial_ls << "\n";
    std::cout << "Final   m: " << sigma << " c: " << ls << "\n";

    return { sigma, ls };
}

int estimate_acf(int argc, char** argv) {

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif


    auto cov = std::make_shared<SquaredExponentialCovariance>(1.0, 1.0f);

    Path basePath = Path("testing/est-acf/1D");
    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    int dim = 2048;
    double scale = 200;

    auto gp = std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), cov);

#if 0
    std::vector<Vec3d> points(dim * dim);
    std::vector<Derivative> derivs(dim * dim, Derivative::None);

    {
        int idx = 0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                points[idx] = scale * (Vec3d((float)i, (float)j, 0.f) / (dim - 1) - 0.5f);
                points[idx][2] = 0.f;
                idx++;
            }
        }
    }
#else
    std::vector<Vec3d> points(dim);
    std::vector<Derivative> derivs(dim, Derivative::None);
    std::vector<double> acf_gt(dim);
    std::vector<double> acf_t(dim);

    {
        int idx = 0;
        for (int i = 0; i < dim; i++) {
            points[idx] = scale * (Vec3d((float)i, 0.f, 0.f) / (dim - 1) - 0.5f);
            acf_gt[idx] = (*cov)(Derivative::None, Derivative::None, points[0], points[idx], Vec3d(), Vec3d());
            acf_t[idx] = (points[0] - points[idx]).length();
            idx++;
        }
    }
#endif


    UniformPathSampler sampler(0);
    sampler.next2D();
    auto basis = WeightSpaceBasis::sample(gp->_cov, 3000, sampler);
    const WeightSpaceRealization ws = basis.sampleRealization(gp, sampler);
    auto sample = ws.evaluate(points.data(), points.size());

    double mean = sample.mean();

    {
        std::ofstream xfile(basePath.asString() + tinyformat::format("/ws-real-%d.bin", basis.size()), std::ios::out | std::ios::binary);
        xfile.write((char*)sample.data(), sizeof(sample[0]) * sample.size());
        xfile.close();
    }

    {
        auto acf = compute_acf_direct(sample.data(), sample.size(), mean);
        std::ofstream xfile(basePath.asString() + "/acf-direct.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)acf.data(), sizeof(acf[0]) * acf.size());
        xfile.close();
    }

    auto acf_fftw = compute_acf_fftw(sample.data(), sample.size(), acf_t[1], mean);
    {
        std::ofstream xfile(basePath.asString() + "/acf-fftw.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)acf_fftw.data(), sizeof(acf_fftw[0]) * acf_fftw.size());
        xfile.close();
    }


    {
        std::ofstream xfile(basePath.asString() + "/acf-gt.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)acf_gt.data(), sizeof(acf_gt[0]) * acf_gt.size());
        xfile.close();
    }

    {
        std::ofstream xfile(basePath.asString() + "/acf-t.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)acf_t.data(), sizeof(acf_t[0]) * acf_t.size());
        xfile.close();
    }



    {
        auto standard_cov = std::make_shared<SquaredExponentialCovariance>();
        auto [sigma, ls] = fit_cov(acf_t.data(), acf_fftw.data(), acf_fftw.size(), standard_cov.get());

        std::vector<double> acf_fit(points.size());
        for (int i = 0; i < points.size(); i++) {
            acf_fit[i] = sqr(sigma) * (*standard_cov)(Derivative::None, Derivative::None, points[0], points[i] / ls , Vec3d(), Vec3d());
        }

        std::ofstream xfile(basePath.asString() + "/acf-fit.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)acf_fit.data(), sizeof(acf_fit[0]) * acf_fit.size());
        xfile.close();
    }
    

#if 0
    auto n = sample.size();

    std::vector<std::complex<double>> spectrumValues(sample.size());
    fftw_plan plan = fftw_plan_dft_r2c_1d(sample.size(), sample.data(), (fftw_complex*)spectrumValues.data(), FFTW_ESTIMATE);

    fftw_execute(plan);
    fftw_destroy_plan(plan);

    std::vector<std::complex<double>> raw_periodogram;
    std::transform(spectrumValues.begin(), spectrumValues.end(), std::back_inserter(raw_periodogram), [n](auto spec) { return sqr(std::abs(spec)) / n; });

    std::vector<double> auto_cov(sample.size());
    plan = fftw_plan_dft_c2r_1d(raw_periodogram.size(), (fftw_complex*)raw_periodogram.data(), (double*)auto_cov.data(), FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
#endif



    /*std::cout << scenePath.parent().asString() << "\n";

    auto bounds = mean.bounds();
    bounds.grow(bounds.diagonal().length() * 0.2);

    Vec3d minp = bounds.min();
    Vec3d maxp = bounds.max();

    Vec3d extends = (maxp - minp);

    double max_extend = extends.max();

    std::cout << extends << ":" << max_extend << "\n";

    Vec3d aspect = extends / max_extend;

    Vec3i dims = vec_conv<Vec3i>(aspect * dim);

    std::cout << aspect << ":" << dims << "\n";

    auto cellSize = max_extend / dim;

    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>(Vec3d(cellSize)));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(minp));

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(100.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("mean");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));
    openvdb::FloatGrid::Accessor meanAccessor = meanGrid->getAccessor();

    UniformPathSampler sampler(0);
    sampler.next2D();

    int num_samples = 1;

    for (int i = 0; i < dims.x(); i++) {
        std::cout << i << "\r";
        for (int j = 0; j < dims.y(); j++) {
#pragma omp parallel for
            for (int k = 0; k < dims.z(); k++) {
                auto p = lerp(minp, maxp, Vec3d((float)i, (float)j, (float)k) / vec_conv<Vec3d>(dims));

                double m = 0;

                if (num_samples > 0) {
                    for (int s = 0; s < num_samples; s++) {
                        Vec2d s1 = rand_normal_2(sampler);
                        Vec2d s2 = rand_normal_2(sampler);
                        Vec3d offset = { (float)s1.x(), (float)s1.y(), (float)s2.x() };
                        Vec3d pt = p + offset * cellSize * 0.1;
                        m += mean(Derivative::None, pt, Vec3d());
                    }
                    m /= num_samples;
                }
                else {
                    m = mean(Derivative::None, p, Vec3d());
                }

#pragma omp critical
                {
                    meanAccessor.setValue({ i,j,k }, m);
                }
            }
        }
    }

    {
        openvdb::GridPtrVec grids;
        grids.push_back(meanGrid);
        openvdb::io::File file(scenePath.stripExtension().asString() + "-" + std::to_string(dim) + ".vdb");
        file.write(grids);
        file.close();
    }*/

}




int main(int argc, char** argv) {

    //mesh_convert(argc, argv);
    //return gen3d(argc, argv);
    //return test2d(argc, argv);

    return estimate_acf(argc, argv);

}
