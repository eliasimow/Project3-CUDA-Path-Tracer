#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>
#include <driver_types.h>

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
    EMPTY,
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
        return intersection.materialId == -1 ? EMPTY : materials[intersection.materialId].materialType; // or whatever your enum field is called
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
    unsigned int left, right;
    unsigned int firstIndex, primCount;
};


