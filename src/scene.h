#pragma once

#include "sceneStructs.h"
#include <vector>
#include <memory>

#include "GltfParse.h"

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
    
    std::vector<glm::vec3> vertPos;

    std::vector<std::unique_ptr<BVHNode>> bvhNodes;

    RenderState state;

    void BufferMesh(std::vector<Mesh> meshes);
};
