#include "SdfFunctions.hpp"
#include <math/Vec.hpp>
#include <math/Mat4f.hpp>
#include <math/MathUtil.hpp>
#include <math/Angle.hpp>
#include <Debug.hpp>

namespace Tungsten {

    std::string SdfFunctions::functionToString(SdfFunctions::Function val)
    {
        switch (val) {
        default:
        case Function::Knob:  return "knob";
        case Function::KnobInner:  return "knob_inner";
        case Function::KnobOuter:  return "knob_outer";
        }
    }

    SdfFunctions::Function SdfFunctions::stringToFunction(const std::string& name)
    {
        if (name == "knob")
            return Function::Knob;
        else if (name == "knob_inner")
            return Function::KnobInner;
        else if (name == "knob_outer")
            return Function::KnobOuter;
        FAIL("Invalid sdf function: '%s'", name);
    }

    /*
    Copyright 2020 Towaki Takikawa @yongyuanxi
    The MIT License
    Link: N/A
    */

    /******************************************************************************
     * The MIT License (MIT)
     * Copyright (c) 2021, NVIDIA CORPORATION.
     * Permission is hereby granted, free of charge, to any person obtaining a copy of
     * this software and associated documentation files (the "Software"), to deal in
     * the Software without restriction, including without limitation the rights to
     * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
     * the Software, and to permit persons to whom the Software is furnished to do so,
     * subject to the following conditions:
     * The above copyright notice and this permission notice shall be included in all
     * copies or substantial portions of the Software.
     * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
     * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
     * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
     * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
     * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
     * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
     ******************************************************************************/

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // distance functions
    // taken from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    float sdSphere(Vec3f v, float r) {
        return v.length() - r;
    }

    float sdTorus(Vec3f p, Vec2f t)
    {
        Vec2f q = Vec2f(p.xz().length() - t.x(), p.y());
        return q.length() - t.y();
    }

    float sdCone(Vec3f p, Vec2f c)
    {
        // c is the sin/cos of the angle
        float q = p.xy().length();
        return c.dot(Vec2f(q, p.z()));
    }

    float sdCappedCylinder(Vec3f p, float h, float r)
    {
        Vec2f d = abs(Vec2f(p.xz().length(), p.y())) - Vec2f(h, r);
        return min(max(d.x(), d.y()), 0.0f) + max(d, Vec2f(0.0f)).length();
    }

    float sdTriPrism(Vec3f p, Vec2f h)
    {
        Vec3f q = abs(p);
        return max(q.z() - h.y(), max(q.x() * 0.866025f + p.y() * 0.5f, -p.y()) - h.x() * 0.5f);
    }

    float opSmoothUnion(float d1, float d2, float k) {
        float h = clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
        return lerp(d2, d1, h) - k * h * (1.0 - h);
    }
    float ssub(float d1, float d2, float k) {
        float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
        return lerp(d2, -d1, h) + k * h * (1.0 - h);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // actual distance functions
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    float sdBase(Vec3f p) {
        // Intersect two cones
        float base = opSmoothUnion(sdCone(Mat4f::rotAxis(Vec3f(1.f, 0.f, 0.f), -90).transformVector(p + Vec3f(0.f, .9f, 0.f)),
            Vec2f(PI / 3., PI / 3.)),
            sdCone(Mat4f::rotAxis(Vec3f(1.f, 0.f, 0.f), 90).transformVector((p - Vec3f(0.f, .9f, 0.f))),
                Vec2f(PI / 3.f, PI / 3.f)),
            0.02);
        // Bound the base radius
        base = max(base, sdCappedCylinder(p, 1.1f, 0.25f)) * 0.7f;
        // Dig out the center
        base = max(-sdCappedCylinder(p, 0.6f, 0.3f), base);
        // Cut a slice of the pie
        base = max(-sdTriPrism(Mat4f::rotAxis(Vec3f(1.f, 0.f, 0.f), 90).transformVector(p + Vec3f(0.f, 0.f, -1.f)), Vec2f(1.2f, 0.3f)), base);
        return base;
    }

    float sdKnob(Vec3f p, int& mat) {
        float sphere = sdSphere(p, 1.0);
        float cutout = sdSphere(p - Vec3f(0.0f, 0.5f, 0.5f), 0.7);
        float cutout_etch = sdTorus(Mat4f::rotAxis(Vec3f(1.f, 0.f, 0.f), -45).transformVector((p - Vec3f(0.0f, 0.2f, 0.2f))), Vec2f(1.0f, 0.05f));
        float innersphere = sdSphere(p - Vec3f(0.0f, 0.0f, 0.0f), 0.75);

        // Cutout sphere
        float d = ssub(cutout, sphere, 0.1);

        // Add eye, etch the sphere
        d = min(d, innersphere);
        d = max(-cutout_etch, d);

        // Add base
        d = min(ssub(sphere,
            sdBase(p - Vec3f(0.f, -.775f, 0.f)), 0.1), d);
        return d;
    }


    float SdfFunctions::knob(Vec3f p, int& mat) {
        const float scale = 0.8;
        p *= 1. / scale;
        return sdKnob(p, mat) * scale;
    }

    float sdKnobInner(Vec3f p, int& mat) {
        return sdSphere(p - Vec3f(0.0f, 0.0f, 0.0f), 0.75);;
    }


    float SdfFunctions::knob_inner(Vec3f p, int& mat) {
        const float scale = 0.8;
        p *= 1. / scale;
        return sdKnobInner(p, mat) * scale;
    }

    float sdKnobOuter(Vec3f p, int& mat) {
        float sphere = sdSphere(p, 1.0);
        float cutout = sdSphere(p - Vec3f(0.0f, 0.5f, 0.5f), 0.7);
        float cutout_etch = sdTorus(Mat4f::rotAxis(Vec3f(1.f, 0.f, 0.f), -45).transformVector((p - Vec3f(0.0f, 0.2f, 0.2f))), Vec2f(1.0f, 0.05f));
        float innersphere = sdSphere(p - Vec3f(0.0f, 0.0f, 0.0f), 0.75);

        // Cutout sphere
        float d = ssub(cutout, sphere, 0.1);

        // Cut out eye, etch the sphere
        d = max(d, -innersphere);
        d = max(-cutout_etch, d);

        // Add base
        d = min(ssub(sphere,
            sdBase(p - Vec3f(0.f, -.775f, 0.f)), 0.1), d);
        return d;
    }

    float SdfFunctions::knob_outer(Vec3f p, int& mat) {
        const float scale = 0.8;
        p *= 1. / scale;
        return sdKnobOuter(p, mat) * scale;
    }
}