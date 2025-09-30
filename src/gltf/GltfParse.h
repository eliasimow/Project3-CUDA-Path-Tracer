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

static const std::unordered_map<std::string, InterpolationType>  interpolationTypeMap = {
    {"LINEAR", LINEAR},
    {"CUBIC", SPLINE},
    {"STEP", STEP}
};

static const std::unordered_map<std::string, TransformType>  transformTypeMap = {
    {"translation", POSITION},
    {"rotation", ROTATION},
    {"scalar", SCALE}
};

void parseTextureFromPath(const std::string& path, int& width, int& height, std::vector<glm::vec4>& texture);
