// we make the ray class
#ifndef RAYH
#define RAYH
#include "vec3.h"
// add device as a prefix to all methods b/c we want this ray class to be only on GPU (and the floor is made out of floor im a dumbass chungus)
class ray
{
public:
    __device__ ray() {}
    __device__ ray(const vec3 &a, const vec3 &b)
    {
        A = a;
        B = b;
    }
    __device__ vec3 origin() const { return A; }
    __device__ vec3 direction() const { return B; }
    __device__ vec3 point_at_parameter(float t) const { return A + t * B; }

    vec3 A;
    vec3 B;
};

#endif