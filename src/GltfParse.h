#pragma once
#include <string>
#include <vector>
#include <optional>
#include <glm/glm.hpp>



struct MaterialInfo {
    std::string name;
    std::optional<std::string> baseColorTexture; // path/uri if available
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
};

struct Mesh {
    std::string name;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords0;
    std::vector<uint32_t> indices;
    std::optional<MaterialInfo> material;
};



class Gltf {

public:
    static std::vector<Mesh> LoadFromFile(const std::string& path);
};