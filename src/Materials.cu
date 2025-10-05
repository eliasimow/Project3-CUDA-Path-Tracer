#include "Materials.cuh"
#include "utilities.h"
#include "interactions.h"
#include "intersections.h"

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

__host__ __device__ void diffuse(PathSegment& path, const ShadeableIntersection& intersection, const Material& mat, int iter, int idx, int depth) {
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
	thrust::uniform_real_distribution<float> u01(0, 1);

	glm::vec3 materialColor = mat.color;
	glm::vec3 newOrigin = path.ray.origin + path.ray.direction * intersection.t + EPSILON * (intersection.surfaceNormal);

	path.ray.origin = newOrigin;
	path.ray.direction = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
	float lightTerm = glm::dot(intersection.surfaceNormal, path.ray.direction);
	path.color *= (materialColor * lightTerm);

	if (glm::dot(path.ray.direction, intersection.surfaceNormal) < 0) {
		path.ray.direction = path.ray.direction * -1.f;
	}
}

__device__ void refract(PathSegment& path, const ShadeableIntersection& intersection, const Material& mat, int iter, int idx, int depth)
{
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

	glm::vec3 hitPoint = path.ray.origin + intersection.t * path.ray.direction;

	glm::vec3 newDir = calculateRefractedDirection(
		path.ray.direction,
		intersection.surfaceNormal,
		mat.indexOfRefraction,
		rng
	);


	//switch to normal push
	bool bouncedOutside = glm::dot(intersection.surfaceNormal, path.ray.direction) > 0.f;
	glm::vec3 newOrigin = hitPoint;

	if (bouncedOutside) {
		newOrigin += intersection.surfaceNormal * FLT_EPSILON * 50.f;
	}
	else {
		newOrigin -= intersection.surfaceNormal * FLT_EPSILON * 50.f;
	}

	path.ray.origin = newOrigin;
	path.ray.direction = newDir;

	path.color *= mat.color;
	path.remainingBounces--;
}

__host__ __device__ void specular(PathSegment& path, const ShadeableIntersection& intersection, const Material& mat, int iter, int idx, int depth)
{
	glm::vec3 materialColor = mat.color;
	glm::vec3 newOrigin = path.ray.origin + path.ray.direction * intersection.t + EPSILON * (intersection.surfaceNormal);

	path.ray.direction = path.ray.direction - 2.f * glm::dot(intersection.surfaceNormal, path.ray.direction) * intersection.surfaceNormal;

	float lightTerm = glm::dot(intersection.surfaceNormal, path.ray.direction);

	path.color *= (materialColor * lightTerm);

	path.ray.origin = newOrigin;

	if (glm::dot(path.ray.direction, intersection.surfaceNormal) < 0) {
		path.ray.direction = path.ray.direction * -1.f;
	}
}

__host__ __device__ void emission(PathSegment& path, const ShadeableIntersection& intersection, const Material& mat, int iter, int idx, int depth)
{
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
	thrust::uniform_real_distribution<float> u01(0, 1);

	glm::vec3 materialColor = mat.color;

	path.color *= (materialColor * mat.emittance);
	path.remainingBounces = 0;
	return;
}

__device__ void environment(PathSegment& path, int iter, int idx, int depth, const cudaTextureObject_t& environmentTexture)
{
	if (depth == 1) {
		path.color = sampleEnvRadiance(environmentTexture, path.ray.direction);
	}
	else {
		path.color *= sampleEnvRadiance(environmentTexture, path.ray.direction);
	}
	path.remainingBounces = 0;
}
