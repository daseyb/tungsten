#ifndef GAUSSIANPROCESSFACTORY_HPP_
#define GAUSSIANPROCESSFACTORY_HPP_

#include "StringableEnum.hpp"

#include <functional>
#include <memory>

namespace Tungsten {

class GaussianProcess;
class CovarianceFunction;
class MeanFunction;

typedef StringableEnum<std::function<std::shared_ptr<GaussianProcess>()>> GaussianProcessFactory;
typedef StringableEnum<std::function<std::shared_ptr<CovarianceFunction>()>> CovarianceFunctionFactory;
typedef StringableEnum<std::function<std::shared_ptr<MeanFunction>()>> MeanFunctionFactory;

}

#endif /* GAUSSIANPROCESSFACTORY_HPP_ */
