#include <math/GPNeuralNetwork.hpp>
#include <core/sampling/UniformPathSampler.hpp>
#include <fstream>
#include <cfloat>
#include <io/Scene.hpp>
#include <thread/ThreadUtils.hpp>

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#endif


using namespace Tungsten;

void nspsr_to_vdb(const GPNeuralNetwork& nn, int dim, Path base_path) {
    ThreadUtils::startThreads(1);

    EmbreeUtil::initDevice();

#ifdef OPENVDB_AVAILABLE
    openvdb::initialize();
#endif

    Vec3d minp = nn.bounds().min();
    Vec3d maxp = nn.bounds().max();

    auto gridTransform = openvdb::Mat4R::identity();
    gridTransform.setToScale(vec_conv<openvdb::Vec3R>((maxp - minp) / dim));
    gridTransform.setTranslation(vec_conv<openvdb::Vec3R>(minp));

    auto meanGrid = openvdb::createGrid<openvdb::FloatGrid>(100.f);
    meanGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    meanGrid->setName("mean");
    meanGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));
    openvdb::FloatGrid::Accessor meanAccessor = meanGrid->getAccessor();

    auto varGrid = openvdb::createGrid<openvdb::FloatGrid>(1.f);
    varGrid->setGridClass(openvdb::GRID_LEVEL_SET);
    varGrid->setName("variance");
    varGrid->setTransform(openvdb::math::Transform::createLinearTransform(gridTransform));
    openvdb::FloatGrid::Accessor varAccessor = varGrid->getAccessor();

    for (int i = 0; i < dim; i++) {
        std::cout << i << "\r";
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                auto p = lerp(minp, maxp, Vec3d((float)i, (float)j, (float)k) / (dim));
                meanAccessor.setValue({ i,j,k }, nn.mean(p));
                varAccessor.setValue({ i,j,k }, sqrt(nn.cov(p, p)));
            }
        }
    }

    {
        openvdb::GridPtrVec grids;
        grids.push_back(meanGrid);
        grids.push_back(varGrid);
        openvdb::io::File file(base_path.stripExtension().asString() + ".vdb");
        file.write(grids);
        file.close();
    }
}

int main() {

    Path file("testing/nspsr/test_3d-network.json");

    std::shared_ptr<JsonDocument> document;
    try {
        document = std::make_shared<JsonDocument>(file);
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
    }

    GPNeuralNetwork network;
    network.read(*document, file.parent());

    std::cout << network.mean(Vec3d(0.1, 0.5, 0.1)) << "\n";
    std::cout << sqrt(network.cov(Vec3d(0.1, 0.5, 0.1), Vec3d(0.1, 0.5, 0.1))) << "\n";

    nspsr_to_vdb(network, 64, file);
}