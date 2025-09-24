#include "utilities.h"
#include "sceneStructs.h"
#include "scene.h"

struct BVHNode
{
    glm::vec3 boxMin, boxMax;
    unsigned int left, right;
    unsigned int firstPrim, primCount;
};

class BVH {
    std::unique_ptr<Triangle[]> tris;
    std::unique_ptr<BVHNode[]> nodes;
    Scene &scene;


};


void UpdateBounds(BVHNode& node, Scene& scene) {

}


void BuildBVH(Scene &scene)
{

    int triCount = scene.vertIdx.size() / 3;
    std::unique_ptr<Triangle[]> tris = std::make_unique<Triangle[]>(
        triCount
    );

    std::unique_ptr<BVHNode[]> nodes = std::make_unique<BVHNode[]>(
        2 * triCount - 1
    );

    unsigned int rootNodeIdx = 0, nodesUsed = 1;



    for each (Geom geo in scene.geoms)
    {
        if (geo.type == MESH) {
            for (int ind = geo.idxStart; ind < geo.idxEnd - 2; ++ind) {
                glm::vec3 centroid = (scene.vertPos[scene.vertIdx[ind]] + scene.vertPos[scene.vertIdx[ind+1]] + scene.vertPos[scene.vertIdx[ind+2]]) * 0.3333f;
                tris[ind] = Triangle();
                tris[ind].indices[0] = scene.vertIdx[ind];
                tris[ind].indices[0] = scene.vertIdx[ind+1];
                tris[ind].indices[0] = scene.vertIdx[ind+2];
                tris[ind].centroid = centroid;
            }
        }
        BVHNode& root = nodes[rootNodeIdx];
        root.left = 0;
        root.right = 0;
        root.firstPrim = 0;
        root.primCount = triCount;

    }
}