#include "GaussianProcessFactory.hpp"

#include "GaussianProcess.hpp"

namespace Tungsten {

DEFINE_STRINGABLE_ENUM(GaussianProcessFactory, "gaussian_process", ({
    {"standard", std::make_shared<GaussianProcess>},
}))


DEFINE_STRINGABLE_ENUM(MeanFunctionFactory, "median", ({
    {"homogeneous", std::make_shared<HomogeneousMean>},
    {"spherical", std::make_shared<SphericalMean>},
}))


DEFINE_STRINGABLE_ENUM(CovarianceFunctionFactory, "covariance", ({
    {"squared_exponential", std::make_shared<SquaredExponentialCovariance>},
}))


}
