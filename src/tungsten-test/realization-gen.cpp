#include <core/media/GaussianProcessMedium.hpp>
#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <tinyformat/tinyformat.hpp>
#include <thread/ThreadUtils.hpp>
#include <io/Scene.hpp>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 32;

std::tuple<std::vector<Vec3d>, std::vector<Vec3d>, std::vector<double>, std::vector<Derivative>> sample_surface(float scale) {

    std::vector<Vec3d> ps;
    std::vector<Vec3d> ns;
    std::vector<double> vs;
    std::vector<Derivative> ds;

    Vec3d c = Vec3d(0.75, 0, 0.f);
    double r = 0.4f;
    /*ps.push_back(c);
    ns.push_back(Vec3d(0.f));
    vs.push_back(-r);
    ds.push_back(Derivative::None);

   ps.push_back(c + r * 2 * Vec3d(-1.f, 1.f, 0.f).normalized());
    ns.push_back(Vec3d(0.f));
    vs.push_back(r);
    ds.push_back(Derivative::None);*/

    int num_pts = 5;
    for (int i = 0; i < num_pts; i++) {
        float a = lerp(PI/2, PI * 1.5f, float(i) / (num_pts-1));
        Vec3d p = c + Vec3d(cos(a), sin(a), 0.f) * r;
        ps.push_back(p*scale);
        ns.push_back((p - c).normalized());
        vs.push_back(0);
        ds.push_back(Derivative::None);

#if 1
        ps.push_back(p * scale);
        ns.push_back((p - c).normalized());
        vs.push_back(1);
        ds.push_back(Derivative::First);
#endif
    }

    return { ps, ns, vs, ds };
}

void gen_cond_test() {

    int num_reals = 2000;
    
    float scale = 10;
    auto gp = std::make_shared<GaussianProcess>(std::make_shared<HomogeneousMean>(), std::make_shared<SquaredExponentialCovariance>(scale, 0.3f * scale));

    std::cout << gp->compute_beckmann_roughness(Vec3d(0.)) << "\n";

    //return;

#if 0
    {
        UniformPathSampler sampler(0);
        sampler.next1D();
        sampler.next1D();


        std::vector<Vec3f> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
        std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

        {
            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    points[idx] = 2.f * (Vec3f((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                    points[idx][2] = 0.f;
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }

            Eigen::MatrixXf samples = gp.sample(
                points.data(), derivs.data(), points.size(), nullptr,
                nullptr, 0,
                Vec3f(1.0f, 0.0f, 0.0f), num_reals, sampler);

            {
                std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-grid-samples-nocond.bin", gp._cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
                xfile.close();
            }
        }
    }
#endif

    auto [cps, cns, cvs, cds] = sample_surface(10.f);

    gp->setConditioning(
        cps,
        cds,
        cns,
        cvs
    );

    {
        std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-cond-ps-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp->_globalCondPs.data(), sizeof(Vec3d) * gp->_globalCondPs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-cond-ds-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp->_globalCondDerivs.data(), sizeof(Derivative) * gp->_globalCondDerivs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-cond-ns-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp->_globalCondDerivDirs.data(), sizeof(Vec3d) * gp->_globalCondDerivDirs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-cond-vs-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp->_globalCondValues.data(), sizeof(double) * gp->_globalCondValues.size());
        xfile.close();
    }

    {
        UniformPathSampler sampler(0);
        sampler.next2D();


        std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
        std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

        {
            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    points[idx] = 2. * (Vec3d((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f) * scale;
                    points[idx][2] = 0.f;
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }

            std::array<Vec3d, 2> ad_cond = {
                Vec3d(0.),
                Vec3d(-0.5f, 0, 0.)
            };

            std::array<Derivative, 2> add_deriv = {
                Derivative::None,
                Derivative::None,
            };

            auto add_v = std::make_shared<GPRealNodeValues>(-0.1 * Eigen::MatrixXd::Ones(2, 1), gp.get());


            auto [samples, gpIds] = gp->sample_cond(
                points.data(), derivs.data(), points.size(), nullptr,
                ad_cond.data(), add_v.get(), add_deriv.data(), 0, nullptr,
                nullptr, 0,
                Vec3d(0.0f, 0.0f, 0.0f), num_reals, sampler)->flatten();

            {
                std::ofstream xfile(tinyformat::format("testing/realizations/%s-%d-grid-samples-cond.bin", gp->_cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
                xfile.close();
            }
        }
    }
}

void gen_real_microfacet_to_volume() {

    UniformPathSampler sampler(0);
    sampler.next2D();


    std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    {
        int idx = 0;
        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                points[idx] = 10. * (Vec3d((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                points[idx][2] = 0.f;
                derivs[idx] = Derivative::None;
                idx++;
            }
        }
    }

    for (auto [meanScale, meanMin] : { std::tuple{1.f, -10000.f}, std::tuple{5.f, -100.f}, std::tuple{100.f, 1.f} }) {
        for (auto [isotropy, lengthScale] : { std::tuple{0.05f, 0.2f}, std::tuple{0.1f, 0.1f}, std::tuple{1.f, 0.05f} }) {
            UniformPathSampler sampler(0);
            sampler.next2D();

            auto gp = std::make_shared<GaussianProcess>(
                std::make_shared<LinearMean>(Vec3d(0., 0., 0.), Vec3d(0., 1., 0.), meanScale, meanMin),
                std::make_shared<SquaredExponentialCovariance>(1.0f, lengthScale, Vec3f(1.f, isotropy, 1.f)));

            auto [samples, gpIds] = gp->sample(
                points.data(), derivs.data(), points.size(), nullptr,
                nullptr, 0,
                Vec3d(0.0f, 0.0f, 0.0f), 1, sampler)->flatten();

            auto path = Path(tinyformat::format("testing/realizations/volume-to-surface/%s-%f-%d-grid-samples-cond.bin", gp->_cov->id(), meanScale, NUM_SAMPLE_POINTS));
            
            if (!path.parent().exists()) {
                FileUtils::createDirectory(path.parent());
            }

            {
                std::ofstream xfile(path.asString(), std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
                xfile.close();
            }
        }
    }


}

void sample_scene_gp(int argc, const char** argv) {

    int dim = 32;

    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    auto scenePath = Path(argv[1]);
    std::cout << scenePath.asString() << "\n";

    Scene* scene = nullptr;
    TraceableScene* tscene = nullptr;
    try {
        scene = Scene::load(scenePath);
        scene->loadResources();
        tscene = scene->makeTraceable();
    }
    catch (std::exception& e) {
        std::cout << e.what();
        return;
    }

    std::shared_ptr<GaussianProcessMedium> gp_medium = std::static_pointer_cast<GaussianProcessMedium>(scene->media()[0]);

    auto gp = std::static_pointer_cast<GPSampleNode>(gp_medium->_gp);

    UniformPathSampler sampler(0);
    sampler.next2D();

    std::vector<Vec3d> points(dim * dim);
    std::vector<Derivative> derivs(dim * dim);
    std::vector<Derivative> fderivs(dim * dim);

    auto processBox = scene->findPrimitive("processBox");

    Vec3d min = vec_conv<Vec3d>(processBox->bounds().min());
    Vec3d max = vec_conv<Vec3d>(processBox->bounds().max());

    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>((max - min) / dim));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(min));

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(100.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("mean");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));

    openvdb::FloatGrid::Accessor meanAccessor = meanGrid->getAccessor();

    {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                int k = dim / 2;
                int idx = i * dim + j ;
                points[idx] = lerp(min, max, Vec3d((float)i, (float)j, (float)k) / (dim));
                derivs[idx] = Derivative::None;
            }
        }
    }

    auto [samples, gpIds] = gp->sample(
        points.data(), derivs.data(), points.size(), nullptr,
        nullptr, 0,
        Vec3d(0.0f, 0.0f, 0.0f), 1, sampler)->flatten();

    auto mean = gp->mean(points.data(), derivs.data(), nullptr, Vec3d(0.), points.size());


    auto path = Path(tinyformat::format("testing/realizations/scene/%s/%s-%d-samples.bin", 
        scenePath.parent().baseName().asString(), 
        scenePath.baseName().stripExtension().asString(), dim));

    if (!path.parent().exists()) {
        FileUtils::createDirectory(path.parent());
    }

    {
        std::ofstream xfile(path.asString(), std::ios::out | std::ios::binary);
        xfile.write((char*)samples.data(), sizeof(double) * samples.rows() * samples.cols());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("testing/realizations/scene/%s/%s-%d-mean.bin",
            scenePath.parent().baseName().asString(),
            scenePath.baseName().stripExtension().asString(), dim), std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(double) * mean.rows() * mean.cols());
        xfile.close();
    }
}

int main(int argc, const char** argv) {
    gen_cond_test();


}
