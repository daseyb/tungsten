#ifndef MEDIUMSAMPLE_HPP_
#define MEDIUMSAMPLE_HPP_

#include "math/Vec.hpp"
#include <Eigen/Dense>

namespace Tungsten {

class PhaseFunction;

struct MediumSample
{
    PhaseFunction *phase;
    Vec3f p;
    float continuedT;
    Vec3f continuedWeight;
    float t;
    Vec3f weight;
    Vec3f emission;
    float pdf;
    bool exited;
    Vec3d aniso;

    Eigen::MatrixXd* usedSamples;
    int usedSampleIdx;
};

}

#endif /* MEDIUMSAMPLE_HPP_ */
