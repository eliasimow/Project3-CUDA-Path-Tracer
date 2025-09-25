#pragma

#include "utilities.h"
#include "sceneStructs.h"
#include "scene.h"

//following https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

class BVH {
public:
    std::vector<int> sortedTriIndices;
    std::vector<std::unique_ptr<BVHNode>> nodes;

    Scene &scene;

    BVH(Scene& scene);
    void BuildBVH();
    void UpdateBounds(BVHNode& node);
    void Subdivide(BVHNode& node);
};