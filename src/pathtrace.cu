#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "BrdsfHelperService.cuh"
#include "BVHNode.cuh"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err)
	{
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file)
	{
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
	getchar();
#endif // _WIN32
	exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y)
	{
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_materialType = NULL;
static int* dev_materialTypeStart = NULL;
static int* dev_materialTypeEnd = NULL;
static VertexData* dev_vertData = NULL;
static Triangle* dev_triangles = NULL;
static BVHNode* dev_BVHNodes = NULL;
static cudaArray_t dev_environmentMap = NULL;
cudaTextureObject_t dev_EnvironmentTexture;

// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));


	//MATERIAL SORTING:

	cudaMalloc(&dev_materialType, pixelcount * sizeof(int));
	cudaMemset(dev_materialType, -1, pixelcount * sizeof(int));


	cudaMalloc(&dev_materialTypeStart, COUNT * sizeof(int));
	cudaMemset(dev_materialTypeStart, -1, COUNT * sizeof(int));


	cudaMalloc(&dev_materialTypeEnd, COUNT * sizeof(int));
	cudaMemset(dev_materialTypeEnd, -1, COUNT * sizeof(int));

	//GLTF:

	if (scene->vertexData.size() > 0) {
		cudaMalloc(&dev_vertData, scene->vertexData.size() * sizeof(VertexData));
		cudaMemcpy(dev_vertData, scene->vertexData.data(), scene->vertexData.size() * sizeof(VertexData), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
		cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_BVHNodes, scene->bvh->nodes.size() * sizeof(BVHNode));
		cudaMemcpy(dev_BVHNodes, scene->bvh->nodes.data(), scene->bvh->nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
	}


	checkCUDAError("pathtraceInit");

	//environment map:
	if (scene->environmentTexture.size() > 0) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		cudaMallocArray(&dev_environmentMap, &channelDesc, scene->environmentWidth, scene->environmentHeight);
		cudaMemcpy2DToArray(dev_environmentMap, 0, 0, scene->environmentTexture.data(), scene->environmentWidth * sizeof(float4), scene->environmentWidth * sizeof(float4), scene->environmentHeight, cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = dev_environmentMap;

		cudaTextureDesc texDesc = {};
		texDesc.addressMode[0] = cudaAddressModeWrap;   // U - wrap so u=0..1 wraps
		texDesc.addressMode[1] = cudaAddressModeClamp;  // V - clamp or clamp_to_edge to avoid seam at poles
		texDesc.filterMode = cudaFilterModeLinear;      // linear filtering
		texDesc.readMode = cudaReadModeElementType;     // return float4
		texDesc.normalizedCoords = 1;                   // coordinates are normalized [0,1]

		dev_EnvironmentTexture = 0;
		cudaCreateTextureObject(&dev_EnvironmentTexture, &resDesc, &texDesc, nullptr);
	}

	checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_materialTypeEnd);
	cudaFree(dev_materialTypeStart);
	cudaFree(dev_materialType);
	cudaFree(dev_triangles);
	cudaFree(dev_vertData);
	cudaFree(dev_BVHNodes);

	checkCUDAError("pathtraceFree");

	cudaDestroyTextureObject(dev_EnvironmentTexture);

	checkCUDAError("pathtraceFree");

	if (dev_environmentMap != nullptr)
		cudaFreeArray(dev_environmentMap);

	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool stochastic)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);



		if (stochastic) {
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, index);
			thrust::uniform_real_distribution<float> range(-.5f, 0.5f);
			// offset
			float moveX = range(rng);
			float moveY = range(rng);

			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x + moveX - (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)y + moveY - (float)cam.resolution.y * 0.5f));
		}
		else {
			// TODO: implement antialiasing by jittering the ray
			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
		}


		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}


// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth,
	int num_paths,
	PathSegment* pathSegments,
	Geom* geoms,
	BVHNode* bvhNodes,
	Triangle* triangles,
	int geoms_size,
	VertexData* vertexData,
	ShadeableIntersection* intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		//todo: compact
		if (pathSegment.remainingBounces == 0) {
			intersections[path_index].materialId = -1;
			return;
		}

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms
		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];
			volatile int type = geom.type;
			type++;
			type--;
			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == TRIANGLES)
			{

				t = intersectBVH(pathSegment.ray, bvhNodes, triangles, vertexData, tmp_intersect, tmp_normal, outside);
			}
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
			intersections[path_index].materialId = -1;
		}
		else
		{
			// The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

__global__ void ShadeDiffuse(
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	int bounceCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < num_paths && pathSegments[idx].remainingBounces > 0)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		// Set up the RNG
		// LOOK: this is how you use thrust's RNG! Please look at
		// makeSeededRandomEngine as well.
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, bounceCount);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;
		glm::vec3 newOrigin = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t + EPSILON * (intersection.surfaceNormal);

		pathSegments[idx].ray.direction = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);

		float lightTerm = glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction);

		pathSegments[idx].color *= (materialColor * lightTerm);

		pathSegments[idx].ray.origin = newOrigin;

		if (glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal) < 0) {
			pathSegments[idx].ray.direction = pathSegments[idx].ray.direction * -1.f;
		}

		//pathSegments[idx].color = (glm::vec3(1., 1., 1.) + intersection.surfaceNormal) / 2.f;
		//pathSegments[idx].remainingBounces = 0;
	}
}

__global__ void ShadeNormal(
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	int bounceCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < num_paths && pathSegments[idx].remainingBounces > 0)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];

		if (intersection.t < 0) {
			pathSegments[idx].color = glm::vec3(0.f);
		}
		else {
			glm::vec3 test = (glm::vec3(1.f) + intersection.surfaceNormal) / 2.f;

			//if (test.x > test.y && test.x > test.z) {
			//	pathSegments[idx].color = glm::normalize(glm::vec3(test.x, 0, 0));
			//}
			//else if (test.y > test.x && test.y > test.z) {
			//	pathSegments[idx].color = glm::normalize(glm::vec3(0, test.y, 0));

			//}
			//else if (test.z > test.x && test.z > test.y) {
			//	pathSegments[idx].color = glm::normalize(glm::vec3(0, 0, test.z));
			//}
			//else {
			//	pathSegments[idx].color = test;

			//}
			pathSegments[idx].color = test;
		}
		pathSegments[idx].remainingBounces = 0;
	}
}

__global__ void ShadeSpecular(
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	int bounceCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < num_paths && pathSegments[idx].remainingBounces > 0)
	{

		ShadeableIntersection intersection = shadeableIntersections[idx];
		// Set up the RNG
		// LOOK: this is how you use thrust's RNG! Please look at
		// makeSeededRandomEngine as well.
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, bounceCount);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;
		glm::vec3 newOrigin = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t + EPSILON * (intersection.surfaceNormal);

		pathSegments[idx].ray.direction = pathSegments[idx].ray.direction - 2.f * glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction) * intersection.surfaceNormal;

		float lightTerm = glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction);

		pathSegments[idx].color *= (materialColor * lightTerm);

		pathSegments[idx].ray.origin = newOrigin;

		if (glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal) < 0) {
			pathSegments[idx].ray.direction = pathSegments[idx].ray.direction * -1.f;
		}
	}
}


__global__ void ShadeEmitting(
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	int bounceCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < num_paths && pathSegments[idx].remainingBounces > 0)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		// Set up the RNG
		// LOOK: this is how you use thrust's RNG! Please look at
		// makeSeededRandomEngine as well.
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, bounceCount);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		pathSegments[idx].color *= (materialColor * material.emittance);
		pathSegments[idx].remainingBounces = 0;
		return;
	}
}


__global__ void ShadeEnvironment(
	int num_paths,
	PathSegment* pathSegments,
	int depth,
	cudaTextureObject_t environmentTexture)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < num_paths && pathSegments[idx].remainingBounces > 0) {
		if (depth == 1) {
			pathSegments[idx].color = sampleEnvRadiance(environmentTexture, pathSegments[idx].ray.direction);
		}
		else {
			pathSegments[idx].color = pathSegments[idx].color * .1f;
		}
		//pathSegments[idx].color *= sampleEnvRadiance(environmentTexture, pathSegments[idx].ray.direction);
		pathSegments[idx].remainingBounces = 0;
	}
}

__global__ void materialIdentifyCellStartEnd(int N, int* materials,
	int* materialStartIndices, int* materialEndIndices) {
	// TODO-2.1
	// Identify the start point of each cell in the gridIndices array.
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= N) {
		return;
	}

	int material = materials[index];
	int materialPrev = index == 0 ? -1 : materials[index - 1];
	if (material != materialPrev) {
		materialStartIndices[material] = index;
		if (materialPrev >= 0) {
			materialEndIndices[materialPrev] = index;
		}
	}

	if (index >= N - 1) {
		materialEndIndices[material] = index + 1;
	}
}


__host__ __device__ bool pathDone(const PathSegment& p) {
	return p.remainingBounces == 0;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter, SceneSettings settings)
{
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, settings.stochastic);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	int alivePaths = num_paths;
	while (depth < traceDepth && alivePaths > 0)
	{
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		cudaMemset(dev_materialTypeEnd, -1, COUNT * sizeof(int));
		cudaMemset(dev_materialTypeStart, -1, COUNT * sizeof(int));

		// tracing
		dim3 numblocksPathSegmentTracing = (alivePaths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth,
			alivePaths,
			dev_paths,
			dev_geoms,
			dev_BVHNodes,
			dev_triangles,
			hst_scene->geoms.size(),
			dev_vertData,
			dev_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

		if (settings.drawNormals) {
			ShadeNormal << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				alivePaths,
				dev_intersections,
				dev_paths,
				dev_materials,
				depth
				);
			continue;
		}


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.


		//PathSegment* mid = thrust::partition(dev_paths, dev_paths + num_paths, HasRemainingBounces());
		//num_paths = static_cast<int>(mid - dev_paths);

		thrust::device_ptr<int> materialEnums(dev_materialType);
		thrust::transform(dev_intersections, dev_intersections + alivePaths, materialEnums,
			MaterialEnumExtractor(dev_materials));

		// Sort both arrays by material enum
		thrust::sort_by_key(materialEnums, materialEnums + alivePaths,
			thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections)));

		materialIdentifyCellStartEnd << < numblocksPathSegmentTracing, blockSize1d >> >
			(
				alivePaths,
				dev_materialType,
				dev_materialTypeStart,
				dev_materialTypeEnd
				);

		std::vector<int> host_start_indices(COUNT);
		std::vector<int> host_end_indices(COUNT);

		cudaMemcpy(host_start_indices.data(), dev_materialTypeStart,
			COUNT * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(host_end_indices.data(), dev_materialTypeEnd,
			COUNT * sizeof(int), cudaMemcpyDeviceToHost);

		//for (int i = 0; i < COUNT; ++i) {
		//    std::printf("\n%d : %d, %d", i, host_start_indices[i], host_end_indices[i]);
		//}

		for (int i = 0; i < COUNT; ++i) {
			MaterialType materialType = MaterialType(i);
			int start = host_start_indices[i];
			int end = host_end_indices[i];
			int count = end - start;

			numblocksPathSegmentTracing = (count + blockSize1d - 1) / blockSize1d;

			if (start >= 0) {
				//std::printf("\nthis is %d start %d end %d", i, start, end);

				switch (materialType)
				{
				case DIFFUSE:
					ShadeDiffuse << <numblocksPathSegmentTracing, blockSize1d >> > (
						iter,
						count,
						dev_intersections + start,
						dev_paths + start,
						dev_materials,
						depth
						);

					checkCUDAError("post pbr write");

					break;
				case SPECULAR:

					ShadeSpecular << <numblocksPathSegmentTracing, blockSize1d >> > (
						iter,
						count,
						dev_intersections + start,
						dev_paths + start,
						dev_materials,
						depth
						);

					break;
				case EMISSION:

					ShadeEmitting << <numblocksPathSegmentTracing, blockSize1d >> > (
						iter,
						count,
						dev_intersections + start,
						dev_paths + start,
						dev_materials,
						depth
						);

					break;
				case PBR:
					break;
				case ENVIRONMENT:
					ShadeEnvironment << <numblocksPathSegmentTracing, blockSize1d >> > (
						count,
						dev_paths + start,
						depth,
						dev_EnvironmentTexture
						);

					checkCUDAError("post env write");

					if (start >= 0 && settings.streamCompact) {
						alivePaths = start;
					}
					break;
				default:
					break;
				}
			}
		}

		checkCUDAError("post material write");


		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}

void rewritePositions(Scene* scene)
{
	//have to resend bvh and positions. i think just that for now:
	cudaMemcpy(dev_vertData, scene->vertexData.data(), scene->vertexData.size() * sizeof(VertexData), cudaMemcpyHostToDevice);

	checkCUDAError("copy animation verts");

	cudaMalloc(&dev_BVHNodes, scene->bvh->nodes.size() * sizeof(BVHNode));
	cudaMemcpy(dev_BVHNodes, scene->bvh->nodes.data(), scene->bvh->nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

	checkCUDAError("copy animation bvh");
}
