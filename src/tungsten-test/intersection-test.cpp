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

void record_first_hit(std::string scene_file, TraceableScene* tracableScene) {

    PathTracerSettings settings;
    settings.enableConsistencyChecks = false;
    settings.enableLightSampling = false;
    settings.enableTwoSidedShading = true;
    settings.enableVolumeLightSampling = false;
    settings.includeSurfaces = true;
    settings.lowOrderScattering = true;
    settings.maxBounces = 5;
    settings.minBounces = 0;

    PathTracer pathTracer(tracableScene, settings, 0);

    int samples = 1000000;
    Eigen::MatrixXf normals(samples, 3);
    Eigen::MatrixXf distanceSamples(samples, 1);
    Eigen::MatrixXf reflectionDirs(samples, 3);

    std::string scene_id = scene_file.find("ref") != std::string::npos ? "surface" : "medium";

    Path basePath = Path("testing/intersections");

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    int sid = 0;


    if (scene_id == "medium") {
        pathTracer._firstMediumBounceCb = [&sid, &normals, &reflectionDirs, &distanceSamples](const MediumSample& mediumSample, Ray r) {
            auto normal = mediumSample.aniso.normalized();
            normals(sid, 0) = normal.x();
            normals(sid, 1) = normal.z();
            normals(sid, 2) = normal.y();

            reflectionDirs(sid, 0) = r.dir().x();
            reflectionDirs(sid, 1) = r.dir().z();
            reflectionDirs(sid, 2) = r.dir().y();

            distanceSamples(sid, 0) = mediumSample.t;

            return false;
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

            return false;
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

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/%s-distances.bin", scene_id)),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)distanceSamples.data(), sizeof(float) * distanceSamples.rows() * distanceSamples.cols());
        xfile.close();
    }
}


void record_paths(std::string scene_file, TraceableScene* tracableScene) {

    PathTracerSettings settings;
    settings.enableConsistencyChecks = false;
    settings.enableLightSampling = false;
    settings.enableTwoSidedShading = true;
    settings.enableVolumeLightSampling = false;
    settings.includeSurfaces = true;
    settings.lowOrderScattering = true;
    settings.maxBounces = 8;
    settings.minBounces = 0;

    PathTracer pathTracer(tracableScene, settings, 0);

    int samples = 1000000;
    Eigen::MatrixXf path_points(samples * settings.maxBounces, 3);
    path_points.setZero();
    

    std::string scene_id = scene_file.find("ref") != std::string::npos ? "surface" : "medium";

    Path basePath = Path("testing/intersections");

    if (!basePath.exists()) {
        FileUtils::createDirectory(basePath);
    }

    int sid = 0;
    int bounce = 0;

    pathTracer._firstMediumBounceCb = [&sid, &path_points, &bounce](const MediumSample& mediumSample, Ray r) {
        path_points.row(sid * 8 + bounce) =  vec_conv<Eigen::Vector3f>(r.pos());
        bounce++;
        return true;
    };
    pathTracer._firstSurfaceBounceCb = [&sid, &path_points, &bounce](const SurfaceScatterEvent& event, Ray r) {
        if (!event.sampledLobe.isForward()) {
            path_points.row(sid * 8 + bounce) = vec_conv<Eigen::Vector3f>(r.pos());
            bounce++;
        }
        return true;
    };


    UniformPathSampler sampler(0);
    sampler.next2D();

    for (sid = 0; sid < samples; sid++) {
        if ((sid + 1) % 100 == 0) {
            std::cout << sid << "/" << samples;
            std::cout << "\r";
        }

        path_points.row(sid * 8) = vec_conv<Eigen::Vector3f>(tracableScene->cam().pos());
        bounce = 1;
        pathTracer.traceSample({ 256, 256 }, sampler);
    }

    {
        std::ofstream xfile(
            incrementalFilename(
                basePath + Path(tinyformat::format("/%s-paths.bin", scene_id)),
                "", false).asString(),
            std::ios::out | std::ios::binary);

        xfile.write((char*)path_points.data(), sizeof(float) * path_points.rows() * path_points.cols());
        xfile.close();
    }
}
 
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

    //record_first_hit(argv[1], tracableScene);

    record_paths(argv[1], tracableScene);

}
