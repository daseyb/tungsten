#include "PhaseFunctionFactory.hpp"

#include "HenyeyGreensteinPhaseFunction.hpp"
#include "IsotropicPhaseFunction.hpp"
#include "RayleighPhaseFunction.hpp"
#include "BRDFPhaseFunction.hpp"

namespace Tungsten {

DEFINE_STRINGABLE_ENUM(PhaseFunctionFactory, "phase function", ({
    {"isotropic", std::make_shared<IsotropicPhaseFunction>},
    {"henyey_greenstein", std::make_shared<HenyeyGreensteinPhaseFunction>},
    {"rayleigh", std::make_shared<RayleighPhaseFunction>},
    {"brdf", std::make_shared<BRDFPhaseFunction>},
}))

}
