#include "BVHNode.cuh"

BVH::BVH(Scene& scene) : scene(scene) {}

void BVH::BuildBVH()
{
    int triCount = scene.triangles.size();

    for (int i = 0; i < triCount; ++i) {
        sortedTriIndices.push_back(i);
    }


    unsigned int rootNodeIdx = 0, nodesUsed = 1;

    //for each (Geom geo in scene.geoms)
    //{
    //    for each (Triangle t in )

    //    if (geo.type == MESH) {
    //        for (int ind = geo.idxStart; ind < geo.idxEnd - 2; ++ind) {
    //            //glm::vec3 centroid = (scene.vertPos[scene.vertIdx[ind]] + scene.vertPos[scene.vertIdx[ind + 1]] + scene.vertPos[scene.vertIdx[ind + 2]]) * 0.3333f;
    //            //tris[ind].centroid = centroid;

    //            //tris[ind] = Triangle();
    //            //tris[ind].vertIndices[0] = scene.vertIdx[ind];
    //            //tris[ind].vertIndices[0] = scene.vertIdx[ind + 1];
    //            //tris[ind].vertIndices[0] = scene.vertIdx[ind + 2];
    //        }
    //    }
    //}

    nodes.push_back(std::make_unique<BVHNode>());
    nodes[0]->left = 0;
    nodes[0]->right = 0;
    nodes[0]->firstIndex = 0;
    nodes[0]->primCount = triCount;    

    UpdateBounds(*nodes[0].get());
    Subdivide(*nodes[0].get());
}

void BVH::UpdateBounds(BVHNode& node) {
    node.boxMin = glm::vec3(1e30f);
    node.boxMax = glm::vec3(-1e30f);
    for (unsigned int first = node.firstIndex, i = 0; i < node.primCount; i++)
    {
        int triIndex = sortedTriIndices[first + i];
        Triangle leafTri = scene.triangles[triIndex];

        node.boxMin = glm::min(node.boxMin, scene.vertPos[leafTri.vertIndices[0]]);
        node.boxMin = glm::min(node.boxMin, scene.vertPos[leafTri.vertIndices[1]]);
        node.boxMin = glm::min(node.boxMin, scene.vertPos[leafTri.vertIndices[2]]);

        node.boxMax = glm::max(node.boxMin, scene.vertPos[leafTri.vertIndices[0]]);
        node.boxMax = glm::max(node.boxMin, scene.vertPos[leafTri.vertIndices[1]]);
        node.boxMax = glm::max(node.boxMin, scene.vertPos[leafTri.vertIndices[2]]);
    }
}

void BVH::Subdivide(BVHNode& node) {
    glm::vec3 diag = node.boxMax - node.boxMin;
    int axis = 0;
    if (diag.y > diag.x) axis = 1;
    if (diag.z > diag[axis]) axis = 2;
    float split = node.boxMin[axis] + diag[axis] * 0.5f;

    int i = node.firstIndex;
    int j = i + node.primCount -1;
    while (i <= j) {
        int triIndex = sortedTriIndices[i];
        Triangle t = scene.triangles[triIndex];
        glm::vec3 centroid = (scene.vertPos[t.vertIndices[0]] + scene.vertPos[t.vertIndices[1]] + scene.vertPos[t.vertIndices[2]]) * 0.3333f;
        if (centroid[axis] < split) {
            i++;
        }
        else {
            std::swap(sortedTriIndices[i], sortedTriIndices[j]);
            j--;
        }
    }

    int leftCount = i - node.firstIndex;
    //no division occured
    if (leftCount == 0 || leftCount == node.primCount) return;

    node.left = nodes.size();
    node.right = nodes.size() + 1;

    //todo need initialize?

    nodes.push_back(std::make_unique<BVHNode>());
    nodes.push_back(std::make_unique<BVHNode>());

    nodes[node.left]->firstIndex = node.firstIndex;
    nodes[node.left]->primCount = leftCount;

    nodes[node.right]->firstIndex = i;
    nodes[node.right]->primCount = node.primCount - leftCount;

    node.primCount = 0;

    UpdateBounds(*nodes[node.left].get());
    UpdateBounds(*nodes[node.left].get());


    if (nodes[node.left]->primCount > 1) {
        Subdivide(*nodes[node.left].get());
    }

    if (nodes[node.right]->primCount > 1) {
        Subdivide(*nodes[node.left].get());
    }
}
