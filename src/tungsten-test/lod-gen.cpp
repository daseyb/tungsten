#include <core/math/GaussianProcess.hpp>
#include <core/media/GaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/Scene.hpp>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#endif

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 128;

int main(int argc, char** argv) {

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

    std::vector<Vec3f> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> fderivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    Vec3f min(-1.0f, 0.0f, -1.0f);
    Vec3f max( 1.0f, 2.0f,  1.0f);

    {
        int idx = 0;
        for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
            for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                for (int k = 0; k < NUM_SAMPLE_POINTS; k++) {
                    points[idx] = lerp(min, max, Vec3f((float)i, (float)j, (float)k) / (NUM_SAMPLE_POINTS - 1));
                    derivs[idx] = Derivative::None;
                    fderivs[idx] = Derivative::First;
                    idx++;
                }
            }
        }
    }

    {
        Eigen::VectorXf mean = gp->mean(points.data(), derivs.data(), Vec3f(1.0f, 0.0f, 0.0f), points.size());
        std::ofstream xfile("./data/testing/load-gen/mean-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)mean.data(), sizeof(float) * mean.rows() * mean.cols());
        xfile.close();
    }

    {
        Eigen::VectorXf gradx = gp->mean(points.data(), fderivs.data(), Vec3f(1.0f, 0.0f, 0.0f), points.size());
        std::ofstream xfile("./data/testing/load-gen/mean-dx-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)gradx.data(), sizeof(float) * gradx.rows() * gradx.cols());
        xfile.close();
    }
    {
        Eigen::VectorXf grady = gp->mean(points.data(), fderivs.data(), Vec3f(0.0f, 1.0f, 0.0f), points.size());
        std::ofstream xfile("./data/testing/load-gen/mean-dy-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)grady.data(), sizeof(float) * grady.rows() * grady.cols());
        xfile.close();
    }
    {
        Eigen::VectorXf gradz = gp->mean(points.data(), fderivs.data(), Vec3f(0.0f, 0.0f, 1.0f), points.size());
        std::ofstream xfile("./data/testing/load-gen/mean-dz-eval.bin", std::ios::out | std::ios::binary);
        xfile.write((char*)gradz.data(), sizeof(float) * gradz.rows() * gradz.cols());
        xfile.close();
    }

    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();



   
}
