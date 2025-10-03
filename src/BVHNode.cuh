#pragma once

#include "utilities.h"
#include "sceneStructs.h"

//following https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

class BVH {
public:
	std::vector<int> sortedTriIndices;
	std::vector<BVHNode> nodes;

	std::vector<Triangle>& triangles;
	std::vector<VertexData>& positions;

	BVH(std::vector<Triangle>& tri, std::vector<VertexData>& pos);
	void BuildBVH();
	void UpdateBounds(int nodeIdx);
	void Subdivide(int nodeIdx);
	void FindBestSplit(int nodeIndex, int& axis, float& splitPosition, float& chosenCost);
	float CalculateNodeCost(int nodeIdx);
	float EvaluateSAH(int nodeIdx, int axis, float pos);
};