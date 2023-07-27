#include <core/math/GaussianProcess.hpp>
#include <core/media/GaussianProcessMedium.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/Scene.hpp>
#include <integrators/path_tracer/PathTracer.hpp>
#include <thread/ThreadUtils.hpp>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif

using namespace Tungsten;

int main(int argc, char** argv) {
    ThreadUtils::startThreads(1);

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

    auto tracableScene = scene->makeTraceable();

    PathTracerSettings settings;
    settings.enableConsistencyChecks = false;
    settings.enableLightSampling = false;
    settings.enableTwoSidedShading = true;
    settings.enableVolumeLightSampling = false;
    settings.includeSurfaces = true;
    settings.lowOrderScattering = true;
    settings.maxBounces = 120;
    settings.minBounces = 0;

    PathTracer pathTracer(tracableScene, settings, 0);
    
    int samples = 1000000;
    Eigen::MatrixXf normals(samples, 3);
    Eigen::MatrixXf reflectionDirs(samples, 3);

    std::string scene_id = std::string(argv[1]).find("ref") != std::string::npos ? "surface" : "medium";

    Path basePath = Path("testing/intersections");

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    int sid = 0;


    if (scene_id == "medium") {
        pathTracer._firstMediumBounceCb = [&sid, &normals, &reflectionDirs](const MediumSample& mediumSample, Ray r) {
            auto normal = mediumSample.aniso.normalized();
            normals(sid, 0) = normal.x();
            normals(sid, 1) = normal.z();
            normals(sid, 2) = normal.y();

            reflectionDirs(sid, 0) = r.dir().x();
            reflectionDirs(sid, 1) = r.dir().z();
            reflectionDirs(sid, 2) = r.dir().y();
        };
    }
    else {
        pathTracer._firstSurfaceBounceCb = [&sid, &normals, &reflectionDirs](const SurfaceScatterEvent& event, Ray r) {
            auto normal = event.info->Ns;
            normals(sid, 0) = normal.x();
            normals(sid, 1) = normal.z();
            normals(sid, 2) = normal.y();

            reflectionDirs(sid, 0) = r.dir().x();
            reflectionDirs(sid, 1) = r.dir().z();
            reflectionDirs(sid, 2) = r.dir().y();
        };
    }
    

    UniformPathSampler sampler(0);
    sampler.next2D();

    for (sid = 0; sid < samples; sid++) {
        if ((sid + 1) % 100 == 0) {
            std::cout << sid << "/" << samples;
            std::cout << "\r";
        }

        pathTracer.traceSample({ 256, 256 }, sampler);
    }

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/%s-normals.bin", scene_id)),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)normals.data(), sizeof(float) * normals.rows() * normals.cols());
        xfile.close();
    }

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/%s-reflection.bin", scene_id)),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)reflectionDirs.data(), sizeof(float) * reflectionDirs.rows() * reflectionDirs.cols());
        xfile.close();
    }

}
