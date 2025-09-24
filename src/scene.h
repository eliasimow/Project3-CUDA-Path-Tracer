#pragma once

#include "sceneStructs.h"
#include <vector>
#include "GltfParse.h"

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;

    std::vector<int> vertIdx;
    std::vector<glm::vec3> vertPos;

    RenderState state;

    void BufferMesh(std::vector<Mesh> meshes);
};
