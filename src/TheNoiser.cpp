#include "TheNoiser.h"
#include <iostream>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

void TheNoiser::init(int width, int height)
{
	optixInit();

	imgWidth = width;
	imgHeight = height;

	CUcontext cuCtx = 0; // current CUDA context
	OptixDeviceContextOptions options = {};
	optixDeviceContextCreate(cuCtx, &options, &context);

	// Create denoiser
	OptixDenoiserOptions denoiserOptions = {};
	denoiserOptions.guideAlbedo = false;  // set true if you have albedo
	denoiserOptions.guideNormal = false;  // set true if you have normals

	optixDenoiserCreate(context, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiserOptions, &denoiser);

	// Compute memory requirements
	OptixDenoiserSizes sizes;
	optixDenoiserComputeMemoryResources(denoiser, imgWidth, imgHeight, &sizes);

	denoiserStateSize = sizes.stateSizeInBytes;
	scratchSize = sizes.withoutOverlapScratchSizeInBytes;

	cudaMalloc(&dev_denoiserState, denoiserStateSize);
	cudaMalloc(&dev_scratch, scratchSize);

	size_t bufferSize = sizeof(glm::vec3) * (imgWidth * imgHeight);

	cudaMalloc(&dev_input, bufferSize);
	cudaMalloc(&dev_output, bufferSize);
	cudaMalloc(&dev_normals, bufferSize);

	// Setup denoiser
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	optixDenoiserSetup(denoiser, stream, imgWidth, imgHeight, reinterpret_cast<CUdeviceptr>(dev_denoiserState), denoiserStateSize, reinterpret_cast<CUdeviceptr>(dev_scratch), scratchSize);
}

std::vector<glm::vec3> TheNoiser::denoise(std::vector<glm::vec3> pixels, std::vector<glm::vec3> normals)
{
	size_t bufferSize = sizeof(glm::vec3) * (imgWidth * imgHeight);
	cudaMemcpy(dev_input, pixels.data(), bufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_normals, normals.data(), bufferSize, cudaMemcpyHostToDevice);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	OptixDenoiserLayer denoiserLayer = {};
	denoiserLayer.input.data = reinterpret_cast<CUdeviceptr>(dev_input);
	denoiserLayer.input.width = imgWidth;
	denoiserLayer.input.height = imgHeight;
	denoiserLayer.input.rowStrideInBytes = imgWidth * sizeof(float3);
	denoiserLayer.input.pixelStrideInBytes = sizeof(float3);
	denoiserLayer.input.format = OPTIX_PIXEL_FORMAT_FLOAT3;

	denoiserLayer.output.data = reinterpret_cast<CUdeviceptr>(dev_output);
	denoiserLayer.output.width = imgWidth;
	denoiserLayer.output.height = imgHeight;
	denoiserLayer.output.rowStrideInBytes = imgWidth * sizeof(float3);
	denoiserLayer.output.pixelStrideInBytes = sizeof(float3);
	denoiserLayer.output.format = OPTIX_PIXEL_FORMAT_FLOAT3;

	OptixDenoiserParams denoiserParams = {};
	denoiserParams.blendFactor = 0.f;

	OptixDenoiserGuideLayer guides = {};
	guides.normal.data = reinterpret_cast<CUdeviceptr>(dev_normals);
	guides.normal.width = imgWidth;
	guides.normal.height = imgHeight;
	guides.normal.rowStrideInBytes = static_cast<unsigned int>(imgWidth * sizeof(glm::vec3));
	guides.normal.pixelStrideInBytes = static_cast<unsigned int>(sizeof(glm::vec3));
	guides.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;

	// Invoke denoiser
	optixDenoiserInvoke(
		denoiser,
		stream,
		&denoiserParams,
		reinterpret_cast<CUdeviceptr>(dev_denoiserState),
		denoiserStateSize,
		&guides,
		&denoiserLayer,
		1,
		0,
		0,
		reinterpret_cast<CUdeviceptr>(dev_scratch),
		scratchSize
	);

	cudaStreamSynchronize(stream);

	std::vector<glm::vec3> denoisedPixels(pixels.size());
	cudaMemcpy(denoisedPixels.data(), dev_output, bufferSize, cudaMemcpyDeviceToHost);

	return denoisedPixels;
}

void TheNoiser::free()
{
	if (dev_denoiserState) cudaFree(dev_denoiserState);
	if (dev_scratch) cudaFree(dev_scratch);
	if (denoiser) optixDenoiserDestroy(denoiser);
	if (context) optixDeviceContextDestroy(context);
	if (dev_input) cudaFree(dev_input);
	if (dev_output) cudaFree(dev_output);
	if (dev_normals) cudaFree(dev_normals);
}
