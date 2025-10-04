#pragma once

#include "sceneStructs.h"
#include <vector>
#include <memory>

#include "gltf/GltfParse.h"
#include "BVHNode.cuh"

class Scene
{
private:
	void loadFromJSON(const std::string& jsonName);
public:
	Scene(std::string filename);

	std::vector<Geom> geoms;
	std::vector<Material> materials;

	//std::vector<unsigned int> vertIdx;
	std::vector<Triangle> triangles;

	//std::vector<glm::vec3> vertPos;
	std::vector<VertexData> vertexData;

	std::vector<glm::vec4> environmentTexture;
	int environmentWidth = 0;
	int environmentHeight = 0;

	int currentFrame = -1;
	int totalFrames;

	static const int fps = 24;
	bool flipGltfNormals = false;


	std::unique_ptr<BVH> bvh;

	FullGltfData gltfData;
	glm::mat4 gltfFrame;

	RenderState state;

	void BuildBVH();

	void BufferMesh(std::vector<Mesh>& meshes, bool flipNormals);

	void IterateFrame();


};
