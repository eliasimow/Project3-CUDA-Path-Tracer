#pragma
#include <cuda.h>
#include <crt/host_defines.h>
#include <glm/glm.hpp>
#include "sceneStructs.h"
#include <thrust/random.h>


__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth);
__host__ __device__ void diffuse(PathSegment& path, const ShadeableIntersection& intersection, const Material& mat, int iter, int idx, int depth);

__device__ void refract(PathSegment& path, const ShadeableIntersection& intersection, const Material& mat, int iter, int idx, int depth);
__host__ __device__ void specular(PathSegment& path, const ShadeableIntersection& intersection, const Material& mat, int iter, int idx, int depth);
__host__ __device__ void emission(PathSegment& path, const ShadeableIntersection& intersection, const Material& mat, int iter, int idx, int depth);
__device__ void environment(PathSegment& path, int iter, int idx, int depth, const cudaTextureObject_t& environmentTexture);