#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"
#include <glm/gtc/quaternion.hpp> 

#include <string>
#include <vector>
#include <driver_types.h>
#include <optional>
#include <unordered_map>


#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLES,
    MESH
};

struct Ray
{
   // Ray(glm::vec3 ori, glm::vec3 dir) : origin(ori), direction(dir){}

    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    //int idxStart = -1;
   // int idxEnd = -1;
};

struct Triangle
{
   // glm::vec3 centroid;
    int vertIndices[3];
   // int indexIndex;
};

enum MaterialType {
    DIFFUSE,
    SPECULAR,
    EMISSION,
    PBR,
    ENVIRONMENT,
    COUNT
};

struct Material
{
    glm::vec3 color;
    MaterialType materialType;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t;
    glm::vec3 surfaceNormal;
    int materialId;
};


struct HasRemainingBounces {
    __device__ bool operator()(const PathSegment& path) const {
        return path.remainingBounces > 0;
    }
};

struct MaterialEnumExtractor {
    Material* materials;  // Pointer to your materials device array

    MaterialEnumExtractor(Material* mats) : materials(mats) {}

    __device__ int operator()(const ShadeableIntersection& intersection) const {
        return intersection.materialId == -1 ? ENVIRONMENT : materials[intersection.materialId].materialType; // or whatever your enum field is called
    }
};


struct BVHNode
{
    BVHNode() {
        boxMin = glm::vec3(0, 0, 0);
        boxMin = glm::vec3(0, 0, 0);
        left = 0;
        right = 0;
        firstIndex = 0;
        primCount = 0;
    }

    glm::vec3 boxMin, boxMax;
    unsigned int left = 0, right = 0;
    unsigned int firstIndex = 0, primCount = 0;
};


//ANIMATION WOO!:

struct Keyframe {
    float time;
    glm::vec3 translation;
    glm::quat rot;
    glm::vec3 scale;
};


enum InterpolationType {
    ROTATIONLINEAR,
    LINEAR,
    STEP,
    SPLINE
};

enum TransformType {
    POSITION,
    SCALE,
    ROTATION
};

struct MaterialInfo {
    std::string name;
    std::optional<std::string> baseColorTexture;
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
};


struct AnimationChannel {
    int targetNode;
    std::vector<float> times;
    std::vector<glm::vec4> values;
    InterpolationType interpolationType;
    TransformType path;

};

struct Animation {
    std::string name;
    std::vector<AnimationChannel> channels;
};

struct Skin {
    std::vector<int> joints;        // node indices of joints
    std::vector<glm::mat4> inverseBindMatrices;
    int skeletonRoot = -1;          // optional
};

struct Node {
    glm::mat4 localMatrix = glm::mat4(1.0f);
    glm::mat4 globalMatrix = glm::mat4(1.0f);
    glm::vec3 translation;
    glm::quat rotation;
    glm::vec3 scale;
    std::vector<int> children;
    int parent = -1;
};

struct Mesh {
    std::string name;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> bindVertPos;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords0;
    std::vector<uint32_t> indices;
    std::vector<glm::uvec4> jointIndices;
    std::vector<glm::vec4>  weights;
    int vertexOffset;
    //std::optional<MaterialInfo> material;
    Skin skin;
};



struct FullGltfData {
    std::vector<Mesh> meshes;
    std::unordered_map<int, Node> nodes;
    std::vector<Animation> animations;
    float animationTime; 

    FullGltfData() {}

    FullGltfData(std::vector<Mesh> meshes,
        std::vector<Node> iNodes,
        std::vector<Animation> animations,
        float animationTime = 0.f)
        : meshes(std::move(meshes)),
        animations(std::move(animations)), 
        animationTime(animationTime) 
    {
    
        for (int i = 0; i < iNodes.size(); ++i) {
            nodes[i] = iNodes[i];
        } 
    }
};