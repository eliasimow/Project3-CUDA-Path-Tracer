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

	cudaMalloc((void**)&dev_denoiserState, denoiserStateSize);
	cudaMalloc((void**)&dev_scratch, scratchSize);

	size_t bufferSize = sizeof(glm::vec3) * (imgWidth * imgHeight);

	cudaMalloc(reinterpret_cast<void**>(&dev_input), bufferSize);
	cudaMalloc(reinterpret_cast<void**>(&dev_output), bufferSize);


	// Setup denoiser
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	optixDenoiserSetup(denoiser, stream, imgWidth, imgHeight, dev_denoiserState, denoiserStateSize, dev_scratch, scratchSize);
}

std::vector<glm::vec3> TheNoiser::denoise(std::vector<glm::vec3> pixels)
{
	size_t bufferSize = sizeof(glm::vec3) * (imgWidth * imgHeight);
	cudaMemcpy(reinterpret_cast<void*>(dev_input), pixels.data(), bufferSize, cudaMemcpyHostToDevice);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Prepare denoiser layer
	OptixDenoiserLayer denoiserLayer = {};
	denoiserLayer.input.data = dev_input; // if you store as CUDA array, convert to ptr
	denoiserLayer.input.width = imgWidth;
	denoiserLayer.input.height = imgHeight;
	denoiserLayer.input.rowStrideInBytes = imgWidth * sizeof(float4);
	denoiserLayer.input.pixelStrideInBytes = sizeof(float4);
	denoiserLayer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

	denoiserLayer.output.data = dev_output;
	denoiserLayer.output.width = imgWidth;
	denoiserLayer.output.height = imgHeight;
	denoiserLayer.output.rowStrideInBytes = imgWidth * sizeof(float4);
	denoiserLayer.output.pixelStrideInBytes = sizeof(float4);
	denoiserLayer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

	// Optional guide layers
	// Denoiser parameters
	OptixDenoiserParams denoiserParams = {};
	denoiserParams.blendFactor = 1.0f;

	OptixDenoiserGuideLayer guides = {};

	// Invoke denoiser
	optixDenoiserInvoke(
		denoiser,
		stream,
		&denoiserParams,
		dev_denoiserState,
		denoiserStateSize,
		&guides,
		&denoiserLayer,
		1,
		0,
		0,
		dev_scratch,
		scratchSize
	);

	cudaStreamSynchronize(stream);

	std::vector<glm::vec3> denoisedPixels(pixels.size());
	cudaMemcpy(denoisedPixels.data(), reinterpret_cast<void*>(dev_output), bufferSize, cudaMemcpyDeviceToHost);

	return denoisedPixels;
}

void TheNoiser::free()
{
	if (dev_denoiserState) cudaFree(&dev_denoiserState);
	if (dev_scratch) cudaFree(&dev_scratch);
	if (denoiser) optixDenoiserDestroy(denoiser);
	if (context) optixDeviceContextDestroy(context);
	if (dev_input) cudaFree(&dev_input);
	if (dev_output) cudaFree(&dev_output);
}
