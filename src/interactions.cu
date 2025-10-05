#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>
#include "BrdsfHelperService.cuh"



__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
	glm::vec3 normal,
	thrust::default_random_engine& rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD)
	{
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
	{
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else
	{
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

__device__ glm::vec3 calculateCosineWeightedDirection(const glm::vec3& normal, thrust::default_random_engine& rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	float r1 = u01(rng);
	float r2 = u01(rng);

	float phi = 2.0f * M_PI * r1;
	float cosTheta = sqrtf(1.0f - r2);
	float sinTheta = sqrtf(r2);

	glm::vec3 localDir = glm::vec3(
		cosf(phi) * sinTheta,
		sinf(phi) * sinTheta,
		cosTheta
	);

	glm::vec3 up = fabs(normal.z) < 0.999f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
	glm::vec3 tangent = glm::normalize(glm::cross(up, normal));
	glm::vec3 bitangent = glm::cross(normal, tangent);

	glm::vec3 worldDir = glm::normalize(
		localDir.x * tangent +
		localDir.y * bitangent +
		localDir.z * normal
	);

	return worldDir;
}



__device__ glm::vec3 calculateRefractedDirection(
	const glm::vec3& rayDirection,
	const glm::vec3& normal,
	float ior,
	thrust::default_random_engine& rng)
{
	glm::vec3 N = normal;
	float cosi = glm::clamp(glm::dot(rayDirection, N), -1.f, 1.f);
	float pureGlass = 1.0f;
	float materialRefraction = ior;

	bool entering = dot(-1.f * rayDirection, normal) > 0;
	if (!entering) {
		N = -N;
		float tmp = pureGlass;
		pureGlass = materialRefraction;
		materialRefraction = tmp;
		cosi = -cosi;
	}

	cosi = fabs(cosi);

	float eta = pureGlass / materialRefraction;
	float k = 1.0f - eta * eta * (1.0f - cosi * cosi);

	thrust::uniform_real_distribution<float> u01(0, 1);
	float R0 = (pureGlass - materialRefraction) * (pureGlass - materialRefraction) / ((pureGlass + materialRefraction) * (pureGlass + materialRefraction));
	float R = R0 + (1.0f - R0) * powf(1.0f - cosi, 5.0f);

	if (k < 0.0f || u01(rng) < R) {
		return glm::normalize(glm::reflect(rayDirection, N));
	}
	else {
		return glm::normalize(eta * rayDirection + (eta * cosi - sqrtf(k)) * N);
	}
}


__host__ __device__ void scatterRay(
	PathSegment& pathSegment,
	glm::vec3 intersect,
	glm::vec3 normal,
	const Material& m,
	thrust::default_random_engine& rng)
{

}