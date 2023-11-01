#ifndef SDFFUNCTIONS_HPP_
#define SDFFUNCTIONS_HPP_

#include <math/Vec.hpp>

namespace Tungsten {
	class SdfFunctions {
		static float knob(Vec3f p);

		template<typename sdf>
		static Vec3f grad(sdf func, Vec3f p) {
			constexpr float eps = 0.001f;

			auto vals = {
				sdf(p + Vec3f(eps, 0.f, 0.f)),
				sdf(p + Vec3f(0.f, eps, 0.f)),
				sdf(p + Vec3f(0.f, 0.f, eps)),
				sdf(p)
			};

			return Vec3f(
				vals[0] - vals[3],
				vals[1] - vals[3],
				vals[2] - vals[3]
			) / eps;
		}
	};

}

#endif SDFFUNCTIONS_HPP_