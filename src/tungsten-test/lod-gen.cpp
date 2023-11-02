#include <core/math/GaussianProcess.hpp>
#include <core/media/GaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/Scene.hpp>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 128;

int gen3d(int argc, char** argv) {

    EmbreeUtil::initDevice();

    std::string prefix = "sphere";

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    Scene* scene = nullptr;
    try {
        scene = Scene::load(Path(argv[1]));
        scene->loadResources();
    }
    catch (std::exception& e) {
        std::cout << e.what();
        return -1;
    }

    std::shared_ptr<GaussianProcessMedium> gp_medium = std::static_pointer_cast<GaussianProcessMedium>(scene->media()[0]);

    auto gp = std::static_pointer_cast<GaussianProcess>(gp_medium->_gp);

    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();

    std::vector<Vec3d> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> fderivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    Vec3d min(-2.0f, 0.0f, -2.0f);
    Vec3d max(2.0f, 4.0f, 2.0f);

    Eigen::VectorXf mean(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    Eigen::VectorXf variance(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    Eigen::VectorXf aniso(6 * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(4.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("density");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(4.0 / NUM_SAMPLE_POINTS));

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
                        Vec3d p = points[idx] + offset * 4.0 / NUM_SAMPLE_POINTS;
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


        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> meanGridSampler(meanGrid->tree(), meanGrid->transform());


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
                            float occupiedSample = meanSample < 0;
                            float occupiedStored = gp->mean(&bp, &derivs[idx], nullptr, Vec3d(1.0f, 0.0f, 0.0f), 1)(0) < 0;
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
        std::ofstream xfile(tinyformat::format("./data/testing/load-gen/%s-mean-eval-avg-%d.bin", prefix, NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(float) * mean.rows() * mean.cols());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("./data/testing/load-gen/%s-var-eval-avg-%d.bin", prefix, NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)variance.data(), sizeof(float) * variance.rows() * variance.cols());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("./data/testing/load-gen/%s-aniso-%d.bin", prefix, NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)aniso.data(), sizeof(float) * aniso.rows() * aniso.cols());
        xfile.close();
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
            openvdb::io::File file(tinyformat::format("./data/testing/load-gen/%s-mean-eval-avg-%d.vdb", prefix, NUM_SAMPLE_POINTS));
            file.write(grids);
            file.close();
        }

        {
            openvdb::GridPtrVec grids;
            grids.push_back(varGrid);
            openvdb::io::File file(tinyformat::format("./data/testing/load-gen/%s-var-eval-avg-%d.vdb", prefix, NUM_SAMPLE_POINTS));
            file.write(grids);
            file.close();
        }

        {
            openvdb::io::File file(tinyformat::format("./data/testing/load-gen/%s-aniso-%d.vdb", prefix, NUM_SAMPLE_POINTS));
            file.write(anisoGrids);
            file.close();
        }
    }

    /*{
        Eigen::VectorXf gradx = gp->mean(points.data(), fderivs.data(), nullptr, Vec3f(1.0f, 0.0f, 0.0f), points.size());
        std::ofstream xfile("./data/testing/load-gen/mean-dx-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)gradx.data(), sizeof(float) * gradx.rows() * gradx.cols());
        xfile.close();
    }
    {
        Eigen::VectorXf grady = gp->mean(points.data(), fderivs.data(), nullptr, Vec3f(0.0f, 1.0f, 0.0f), points.size());
        std::ofstream xfile("./data/testing/load-gen/mean-dy-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)grady.data(), sizeof(float) * grady.rows() * grady.cols());
        xfile.close();
    }
    {
        Eigen::VectorXf gradz = gp->mean(points.data(), fderivs.data(), nullptr, Vec3f(0.0f, 0.0f, 1.0f), points.size());
        std::ofstream xfile("./data/testing/load-gen/mean-dz-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)gradz.data(), sizeof(float) * gradz.rows() * gradz.cols());
        xfile.close();
    }*/


    return 0;

}

int test2d(int argc, char** argv) {
    return 0;
}

int main(int argc, char** argv) {

    return gen3d(argc, argv);
    //return test2d(argc, argv);

}
