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

constexpr size_t NUM_SAMPLE_POINTS = 256;

int gen3d(int argc, char** argv) {

    EmbreeUtil::initDevice();

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

    auto gp = gp_medium->_gp;


    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();

    std::vector<Vec3f> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> fderivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    Vec3f min(-2.0f, 0.0f, -2.0f);
    Vec3f max(2.0f, 4.0f, 2.0f);

    Eigen::VectorXf mean(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    Eigen::VectorXf variance(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
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
                    points[idx] = lerp(min, max, Vec3f((float)i, (float)j, (float)k) / (NUM_SAMPLE_POINTS - 1));
                    derivs[idx] = Derivative::None;
                    fderivs[idx] = Derivative::First;

                    mean[idx] = 0;
                    std::vector<float> samples(numEstSamples);
                    for (int s = 0; s < numEstSamples; s++) {
                        Vec2d s1 = gp->rand_normal_2(sampler);
                        Vec2d s2 = gp->rand_normal_2(sampler);
                        Vec3f offset = { (float)s1.x(), (float)s1.y(), (float)s2.x() };
                        Vec3f p = points[idx] + offset * 4.0f / NUM_SAMPLE_POINTS;
                        samples[s] = gp->mean(&p, &derivs[idx], nullptr, Vec3f(1.0f, 0.0f, 0.0f), 1)(0);
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
                    Vec3f cp = Vec3f((float)i, (float)j, (float)k);

                    std::vector<float> samples(numEstSamples);
                    int valid_samples = 0;
                    for (int s = 0; s < numEstSamples; s++) {
                        Vec2d s1 = gp->rand_normal_2(sampler);
                        Vec2d s2 = gp->rand_normal_2(sampler);
                        Vec3f offset = { (float)s1.x(), (float)s1.y(), (float)s2.x() };
                        Vec3f p = cp + offset * 4;

                        if (p.x() < 0 || p.y() < 0 || p.z() < 0 ||
                            p.x() > NUM_SAMPLE_POINTS-1 || p.y() > NUM_SAMPLE_POINTS - 1 || p.z() > NUM_SAMPLE_POINTS - 1) {
                            samples[s] = 0;
                        }
                        else {
                            float meanSample = meanGridSampler.isSample(openvdb::Vec3R(p.x(), p.y(), p.z()));
                            Vec3f bp = lerp(min, max, p / (NUM_SAMPLE_POINTS - 1));
                            float occupiedSample = meanSample < 0;
                            float occupiedStored = gp->mean(&bp, &derivs[idx], nullptr, Vec3f(1.0f, 0.0f, 0.0f), 1)(0) < 0;
                            samples[s] = occupiedSample - occupiedStored;
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
                }
            }
        }
    }

    {
        std::ofstream xfile(tinyformat::format("./data/testing/load-gen/tree-voxelized-mean-eval-avg-%d.bin", NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(float) * mean.rows() * mean.cols());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("./data/testing/load-gen/tree-voxelized-var-eval-avg-%d.bin", NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)variance.data(), sizeof(float) * variance.rows() * variance.cols());
        xfile.close();
    }

    {


        auto varGrid = openvdb::createGrid<openvdb::FloatGrid>();
        openvdb::FloatGrid::Accessor varAccessor = varGrid->getAccessor();
        varGrid->setName("density");
        varGrid->setTransform(openvdb::math::Transform::createLinearTransform(4.0 / NUM_SAMPLE_POINTS));

        int idx = 0;
        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            std::cout << i << "\r";
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    varAccessor.setValue({ i,j,k }, variance(idx));
                    idx++;
                }
            }
        }

        {
            openvdb::GridPtrVec grids;
            grids.push_back(meanGrid);
            openvdb::io::File file(tinyformat::format("./data/testing/load-gen/tree-voxelized-mean-eval-avg-%d.vdb", NUM_SAMPLE_POINTS));
            file.write(grids);
            file.close();
        }

        {
            openvdb::GridPtrVec grids;
            grids.push_back(varGrid);
            openvdb::io::File file(tinyformat::format("./data/testing/load-gen/tree-voxelized-var-eval-avg-%d.vdb", NUM_SAMPLE_POINTS));
            file.write(grids);
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
