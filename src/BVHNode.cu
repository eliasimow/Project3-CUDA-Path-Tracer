#include "BVHNode.cuh"



BVH::BVH(std::vector<Triangle>& tri, std::vector<glm::vec3>& pos) : triangles(tri), positions(pos)
{}

void BVH::BuildBVH()
{
    int triCount = triangles.size();

    for (int i = 0; i < triCount; ++i) {
        sortedTriIndices.push_back(i);
    }

    unsigned int rootNodeIdx = 0, nodesUsed = 1;

    nodes.push_back(BVHNode());
    nodes[0].primCount = triCount;

    UpdateBounds(nodes[0]);
    Subdivide(0);
}

void BVH::UpdateBounds(BVHNode& node) {
    node.boxMin = glm::vec3(1e30f);
    node.boxMax = glm::vec3(-1e30f);
    for (unsigned int first = node.firstIndex, i = 0; i < node.primCount; i++)
    {
        int triIndex = sortedTriIndices[first + i];
        Triangle leafTri = triangles[triIndex];

        node.boxMin = glm::min(node.boxMin, positions[leafTri.vertIndices[0]]);
        node.boxMin = glm::min(node.boxMin, positions[leafTri.vertIndices[1]]);
        node.boxMin = glm::min(node.boxMin, positions[leafTri.vertIndices[2]]);

        node.boxMax = glm::max(node.boxMax, positions[leafTri.vertIndices[0]]);
        node.boxMax = glm::max(node.boxMax, positions[leafTri.vertIndices[1]]);
        node.boxMax = glm::max(node.boxMax, positions[leafTri.vertIndices[2]]);
    }
}

void BVH::Subdivide(int nodeIdx) {
    glm::vec3 diag = nodes[nodeIdx].boxMax - nodes[nodeIdx].boxMin;
    int axis = 0;
    if (diag.y > diag.x) axis = 1;
    if (diag.z > diag[axis]) axis = 2;
    float split = nodes[nodeIdx].boxMin[axis] + diag[axis] * 0.5f;

    int i = nodes[nodeIdx].firstIndex;
    int j = i + nodes[nodeIdx].primCount -1;
    while (i <= j) {
        int triIndex = sortedTriIndices[i];
        Triangle t = triangles[triIndex];
        glm::vec3 centroid = (positions[t.vertIndices[0]] + positions[t.vertIndices[1]] + positions[t.vertIndices[2]]) * 0.3333f;
        if (centroid[axis] < split) {
            i++;
        }
        else {
            std::swap(sortedTriIndices[i], sortedTriIndices[j]);
            j--;
        }
    }

    int leftCount = i - nodes[nodeIdx].firstIndex;
    //no division occured
    if (leftCount == 0 || leftCount == nodes[nodeIdx].primCount) return;

    nodes[nodeIdx].left = nodes.size();
    nodes[nodeIdx].right = nodes.size() + 1;

    //todo need initialize?

    nodes.push_back(BVHNode());
    nodes.push_back(BVHNode());

    nodes[nodes[nodeIdx].left].firstIndex = nodes[nodeIdx].firstIndex;
    nodes[nodes[nodeIdx].left].primCount = leftCount;

    nodes[nodes[nodeIdx].right].firstIndex = i;
    nodes[nodes[nodeIdx].right].primCount = nodes[nodeIdx].primCount - leftCount;

    nodes[nodeIdx].primCount = 0;

    UpdateBounds(nodes[nodes[nodeIdx].left]);
    UpdateBounds(nodes[nodes[nodeIdx].right]);


    if (nodes[nodes[nodeIdx].left].primCount > 1) {
        Subdivide(nodes[nodeIdx].left);
    }

    if (nodes[nodes[nodeIdx].right].primCount > 1) {
        Subdivide(nodes[nodeIdx].right);
    }
}
