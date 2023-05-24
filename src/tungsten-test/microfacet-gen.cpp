#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/media/GaussianProcessMedium.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <io/ImageIO.hpp>
#include <io/FileUtils.hpp>
#include <bsdfs/Microfacet.hpp>
#include <rapidjson/document.h>
#include <io/JsonDocument.hpp>
#include <io/JsonObject.hpp>
#include <io/Scene.hpp>
#include <math/GaussianProcessFactory.hpp>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 64;

size_t gidx(int i, int j) {
    return i * NUM_SAMPLE_POINTS + j;
}

Eigen::MatrixXf compute_normals(const Eigen::MatrixXf& samples) {

    Eigen::MatrixXf normals(3, NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    normals.setZero();

    auto samplesr = samples.reshaped(NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS);

    for (int i = 1; i < NUM_SAMPLE_POINTS-1; i++) {
        for (int j = 1; j < NUM_SAMPLE_POINTS-1; j++) {

            float eps = 2.f / NUM_SAMPLE_POINTS;
            
            auto r = samplesr(i + 1, j);
            auto l = samplesr(i - 1, j);
            auto b = samplesr(i, j + 1);
            auto t = samplesr(i, j - 1);

            Vec3f norm = Vec3f(-(r - l) / (2*eps), (b - t) / (2*eps), 1.f).normalized();

            normals(0, gidx(j, i)) = norm.x();
            normals(1, gidx(j, i)) = norm.y();
            normals(2, gidx(j, i)) = norm.z();
        }
    }

    return normals;
}

float compute_beckmann_roughness(const CovarianceFunction& cov) {
    float L2 = cov(Derivative::First, Derivative::First, Vec3f(0.f), Vec3f(0.f), Vec3f(1.f, 0.f, 0.f), Vec3f(1.f, 0.f, 0.f));
    return sqrt(2 * L2);
}


void normals_and_stuff(const GaussianProcess& gp) {
    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();


    std::vector<Vec3f> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    {
        {
            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    points[idx] = 2.f * (Vec3f((float)i, (float)j, 0.f) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }
        }


        Eigen::MatrixXf samples = gp.sample(
            points.data(), derivs.data(), points.size(), nullptr,
            nullptr, 0,
            Vec3f(1.0f, 0.0f, 0.0f), 100, sampler).cast<float>();

        std::cout << samples.minCoeff() << "-" << samples.maxCoeff() << std::endl;


        Eigen::MatrixXf thresholded(samples.rows(), samples.cols());
        {
            for (int i = 0; i < samples.cols(); i++) {
                for (int j = 0; j < samples.rows(); j++) {
                    thresholded(j, i) = samples(j, i) < 0 ? 0 : 1;
                }
            }
        }

        for (int i = 0; i < samples.cols(); i++) {
            Eigen::MatrixXf normals = compute_normals(samples.col(i));
            normals.array() += 1.0f;
            normals.array() *= 0.5f;

            samples.col(i).array() -= samples.col(i).minCoeff();
            samples.col(i).array() /= samples.col(i).maxCoeff();

            ImageIO::saveHdr(incrementalFilename("microfacet/normals/squared-exponential-0.5/rel.exr", "", false), samples.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
            ImageIO::saveHdr(incrementalFilename("microfacet/normals/squared-exponential-0.5/thr.exr", "", false), thresholded.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
            ImageIO::saveHdr(incrementalFilename("microfacet/normals/squared-exponential-0.5/normals.exr", "", false), normals.data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 3);
        }
    }
}

void side_view(const GaussianProcess& gp, std::string output) {
    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();

    std::vector<Vec3f> points(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
    std::vector<Derivative> derivs(NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);

    {
        {
            int idx = 0;
            for (int i = 0; i < NUM_SAMPLE_POINTS; i++) {
                for (int j = 0; j < NUM_SAMPLE_POINTS; j++) {
                    points[idx] = 20.f * (Vec3f((float)j, 0.f, (float)i) / (NUM_SAMPLE_POINTS - 1) - 0.5f);
                    derivs[idx] = Derivative::None;
                    idx++;
                }
            }
        }


        Eigen::MatrixXf samples = gp.sample(
            points.data(), derivs.data(), points.size(), nullptr,
            nullptr, 0,
            Vec3f(1.0f, 0.0f, 0.0f), 1, sampler).cast<float>();

        std::cout << samples.minCoeff() << "-" << samples.maxCoeff() << std::endl;


        Eigen::MatrixXf thresholded(samples.rows(), samples.cols());
        {
            for (int i = 0; i < samples.cols(); i++) {
                for (int j = 0; j < samples.rows(); j++) {
                    thresholded(j, i) = samples(j, i) < 0 ? 0 : 1;
                }
            }
        }

        for (int i = 0; i < samples.cols(); i++) {
            Path basePath = Path(output) / Path(gp._cov->id());

            std::ofstream xfile(incrementalFilename(basePath + Path("-rel.bin"), "", false).asString(), std::ios::out | std::ios::binary);
            xfile.write((char*)samples.col(i).data(), sizeof(float) * NUM_SAMPLE_POINTS * NUM_SAMPLE_POINTS);
            xfile.close();

            samples.col(i).array() -= samples.col(i).minCoeff();
            samples.col(i).array() /= samples.col(i).maxCoeff();

            ImageIO::saveHdr(incrementalFilename(basePath + Path("-rel.exr"), "", false), samples.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
            ImageIO::saveHdr(incrementalFilename(basePath + Path("-thr.exr"), "", false), thresholded.col(i).data(), NUM_SAMPLE_POINTS, NUM_SAMPLE_POINTS, 1);
        }
    }
}


constexpr size_t NUM_RAY_SAMPLE_POINTS = 128;


void sample_beckmann(float alpha) {

    UniformPathSampler sampler(0);
    sampler.next2D();

    Eigen::MatrixXf normals(10000000, 3);
    Microfacet::Distribution distribution("beckmann");

    for (int i = 0; i < normals.rows(); i++) {
        Vec3f normal = Microfacet::sample(distribution, alpha, sampler.next2D());
        normals(i, 0) = normal[0];
        normals(i, 1) = normal[1];
        normals(i, 2) = normal[2];
    }

    {
        std::ofstream xfile(incrementalFilename(Path(tinyformat::format("microfacet/visible-normals/beckmann-%.4f.bin", alpha)), "", false).asString(), std::ios::out | std::ios::binary);
        xfile.write((char*)normals.data(), sizeof(float) * normals.rows() * normals.cols());
        xfile.close();
    }

}

void v_ndf(std::shared_ptr<GaussianProcess> gp, float angle, int samples, std::string output) {

    auto gp_med = std::make_shared<GaussianProcessMedium>(gp, 0, 1, 1, NUM_RAY_SAMPLE_POINTS);

    UniformPathSampler sampler(0);
    sampler.next1D();
    sampler.next1D();

    Ray ray = Ray(Vec3f(0.f, 0.f, 50.f), Vec3f(sin(angle), 0.f, -cos(angle)));

    ray.setNearT(-(ray.pos().z()-5.0f) / ray.dir().z());
    ray.setFarT(-(ray.pos().z()+5.0f) / ray.dir().z());

    Eigen::MatrixXf normals(samples, 3);

    int failed = 0;

    for (int s = 0; s < samples;) {

        if ((s + 1) % 100 == 0) {
            std::cout << s << "/" << samples << " - Failed: " << failed;
            std::cout << "\r";
        }

        Medium::MediumState state;
        state.reset();
        MediumSample sample;
        if (!gp_med->sampleDistance(sampler, ray, state, sample)) {
            failed++;
            continue;
        }

        sample.aniso.normalize();
        normals(s, 0) = sample.aniso.x();
        normals(s, 1) = sample.aniso.y();
        normals(s, 2) = sample.aniso.z();

        s++;
    }

    {
        std::ofstream xfile(incrementalFilename(Path(output) / Path(gp->_cov->id()) + Path(tinyformat::format("-%.1fdeg-%d.bin", 180 * angle / PI, NUM_RAY_SAMPLE_POINTS)), "", false).asString(), std::ios::out | std::ios::binary);
        xfile.write((char*)normals.data(), sizeof(float) * normals.rows() * normals.cols());
        xfile.close();
    }
}

template<typename T>
static std::shared_ptr<T> instantiate(JsonPtr value, const Scene& scene)
{
    auto result = StringableEnum<std::function<std::shared_ptr<T>()>>(value.getRequiredMember("type")).toEnum()();
    result->fromJson(value, scene);
    return result;
}

int main(int argc, char** argv) {
    auto lmean = std::make_shared<LinearMean>(Vec3f(0.f), Vec3f(0.f, 0.f, 1.f), 1.0f);

    std::shared_ptr<JsonDocument> document;
    try {
        document = std::make_shared<JsonDocument>(argc > 1 ? argv[1] : "microfacet/covariances-paper.json");
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
        return -1;
    }

    float aniso = 1.0;
    float angle =  0.;
    Scene scene;

    auto covs = (*document)["covariances"];
    if (!covs || !covs.isArray()) {
        std::cerr << "There should be a `covariances` array in the file\n";
        return -1;
    }

    for (int i = 0; i < covs.size(); i++) {
        const auto& jcov = covs[i];
        try {
            auto cov = instantiate<CovarianceFunction>(jcov, scene);
            cov->loadResources();
            //cov->_aniso[2] = aniso;

            std::cout << cov->id() << "\n";

            auto gp = std::make_shared<GaussianProcess>(lmean, cov);
            gp->_covEps = 0; // 0.00001f;
            gp->_maxEigenvaluesN = 1024;

            float alpha = compute_beckmann_roughness(*gp->_cov);
            std::cout << "Beckmann roughness: " << alpha << "\n";

            sample_beckmann(alpha);

            auto testFile = Path("microfacet/visible-normals/") / Path(gp->_cov->id()) + Path(tinyformat::format("-%.1fdeg-%d.bin", 180 * angle / PI, NUM_RAY_SAMPLE_POINTS));
            if (testFile.exists()) {
                std::cout << "skipping...\n";
                continue;
            }

            v_ndf(gp, angle, 10000, "microfacet/visible-normals/");
            //side_view(gp, "microfacet/side-view/");
        }
        catch (std::exception& e) {
            std::cerr << e.what() << "\n";
        }
    }
}


