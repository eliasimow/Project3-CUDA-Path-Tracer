#pragma

#include <optix.h>
#include <cuda_runtime.h>
#include <vector>
#include "glm/glm.hpp"


class TheNoiser
{
public:
	TheNoiser() = default;

	void init(int width, int height);
	std::vector<glm::vec3> denoise(std::vector<glm::vec3> beautyPic, std::vector<glm::vec3> normals);
	void free();

private:
	OptixDeviceContext context = nullptr;
	OptixDenoiser denoiser = nullptr;

	void* dev_denoiserState = 0;
	size_t denoiserStateSize = 0;

	void* dev_scratch = 0;
	size_t scratchSize = 0;

	void* dev_input = 0;
	void* dev_normals = 0;
	void* dev_output = 0;


	int imgWidth = 0;
	int imgHeight = 0;
};