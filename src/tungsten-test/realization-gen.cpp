#include <core/math/GaussianProcess.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <core/math/Ray.hpp>
#include <fstream>
#include <cfloat>
#include <tinyformat/tinyformat.hpp>

using namespace Tungsten;

constexpr size_t NUM_SAMPLE_POINTS = 32;

std::tuple<std::vector<Vec3f>, std::vector<Vec3f>, std::vector<float>, std::vector<Derivative>> sample_surface() {

    std::vector<Vec3f> ps;
    std::vector<Vec3f> ns;
    std::vector<float> vs;
    std::vector<Derivative> ds;

    Vec3f c = Vec3f(0.75, -0.75, 0.f);
    float r = 0.5f;
    /*ps.push_back(c);
    ns.push_back(Vec3f(0.f));
    vs.push_back(-r);
    ds.push_back(Derivative::None);

    ps.push_back(c + r * 2 * Vec3f(-1.f, 1.f, 0.f).normalized());
    ns.push_back(Vec3f(0.f));
    vs.push_back(r);
    ds.push_back(Derivative::None);*/

    for (int i = 0; i < 10; i++) {
        float a = lerp(PI / 2 - 0.1f, PI + 0.1f, float(i) / 9);
        Vec3f p = c + Vec3f(cos(a), sin(a), 0.f) * 0.5f;
        ps.push_back(p);
        ns.push_back((p - c).normalized());
        vs.push_back(0);
        ds.push_back(Derivative::None);

        ps.push_back(p);
        ns.push_back((p - c).normalized());
        vs.push_back(-1);
        ds.push_back(Derivative::First);
    }

    return { ps, ns, vs, ds };
}

int main() {

    int num_reals = 2000;

	GaussianProcess gp(std::make_shared<HomogeneousMean>(), std::make_shared<SquaredExponentialCovariance>(1.0f, 0.25f));
    
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
                std::ofstream xfile(tinyformat::format("realizations/%s-%d-grid-samples-nocond.bin", gp._cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
                xfile.close();
            }
        }
    }
#endif

    auto [cps, cns, cvs, cds] = sample_surface();

    gp.setConditioning(
        cps,
        cds,
        cns,
        cvs
    );
    
    {
        std::ofstream xfile(tinyformat::format("realizations/%s-%d-cond-ps-cond.bin", gp._cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp._globalCondPs.data(), sizeof(Vec3f) * gp._globalCondPs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("realizations/%s-%d-cond-ds-cond.bin", gp._cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp._globalCondDerivs.data(), sizeof(Derivative) * gp._globalCondDerivs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("realizations/%s-%d-cond-ns-cond.bin", gp._cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp._globalCondDerivDirs.data(), sizeof(Vec3f) * gp._globalCondDerivDirs.size());
        xfile.close();
    }

    {
        std::ofstream xfile(tinyformat::format("realizations/%s-%d-cond-vs-cond.bin", gp._cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
        xfile.write((char*)gp._globalCondValues.data(), sizeof(float) * gp._globalCondValues.size());
        xfile.close();
    }

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
                Vec3f(0.0f, 0.0f, 0.0f), num_reals, sampler);

            {
                std::ofstream xfile(tinyformat::format("realizations/%s-%d-grid-samples-cond.bin", gp._cov->id(), NUM_SAMPLE_POINTS), std::ios::out | std::ios::binary);
                xfile.write((char*)samples.data(), sizeof(float) * samples.rows() * samples.cols());
                xfile.close();
            }
        }
    }
   
}
