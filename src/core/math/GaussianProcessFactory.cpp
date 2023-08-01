#include "GaussianProcessFactory.hpp"

#include "GaussianProcess.hpp"

namespace Tungsten {

DEFINE_STRINGABLE_ENUM(GaussianProcessFactory, "gaussian_process", ({
    {"standard", std::make_shared<GaussianProcess>},
}))


DEFINE_STRINGABLE_ENUM(MeanFunctionFactory, "mean", ({
    {"homogeneous", std::make_shared<HomogeneousMean>},
    {"spherical", std::make_shared<SphericalMean>},
    {"linear", std::make_shared<LinearMean>},
    {"tabulated", std::make_shared<TabulatedMean>},
    {"mesh", std::make_shared<MeshSdfMean>},
}))


DEFINE_STRINGABLE_ENUM(CovarianceFunctionFactory, "covariance", ({
    {"squared_exponential", std::make_shared<SquaredExponentialCovariance>},
    {"rational_quadratic", std::make_shared<RationalQuadraticCovariance>},
    {"periodic", std::make_shared<PeriodicCovariance>},
    {"nonstationary", std::make_shared<NonstationaryCovariance>},
    {"mg-nonstationary", std::make_shared<MeanGradNonstationaryCovariance>},
    {"thin_plate", std::make_shared<ThinPlateCovariance>},
}))


}
