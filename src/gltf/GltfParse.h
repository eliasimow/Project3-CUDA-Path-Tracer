#pragma once
#include <string>
#include <vector>
#include <optional>
#include <glm/glm.hpp>
#include "../sceneStructs.h"


class Gltf {

public:
    static FullGltfData LoadFromFile(const std::string& path);
};

void parseTextureFromPath(const std::string& path, int& width, int& height, std::vector<glm::vec4>& texture);
