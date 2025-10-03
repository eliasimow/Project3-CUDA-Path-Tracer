#include "BVHNode.cuh"



BVH::BVH(std::vector<Triangle>& tri, std::vector<VertexData>& pos) : triangles(tri), positions(pos)
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

	UpdateBounds(0);
	Subdivide(0);
}

void BVH::UpdateBounds(int nodeIdx) {
	BVHNode& node = nodes[nodeIdx];
	node.boxMin = glm::vec3(1e30f);
	node.boxMax = glm::vec3(-1e30f);
	for (unsigned int first = node.leftOrFirstTri, i = 0; i < node.primCount; i++)
	{
		int triIndex = sortedTriIndices[first + i];
		Triangle& leafTri = triangles[triIndex];

		node.boxMin = glm::min(node.boxMin, positions[leafTri.vertIndices[0]].position);
		node.boxMin = glm::min(node.boxMin, positions[leafTri.vertIndices[1]].position);
		node.boxMin = glm::min(node.boxMin, positions[leafTri.vertIndices[2]].position);

		node.boxMax = glm::max(node.boxMax, positions[leafTri.vertIndices[0]].position);
		node.boxMax = glm::max(node.boxMax, positions[leafTri.vertIndices[1]].position);
		node.boxMax = glm::max(node.boxMax, positions[leafTri.vertIndices[2]].position);
	}
}

float BVH::EvaluateSAH(int nodeIdx, int axis, float pos) {
	aabb leftBox, rightBox;
	int leftCount = 0, rightCount = 0;
	for (int i = 0; i < nodes[nodeIdx].primCount; ++i)
	{
		Triangle& t = triangles[sortedTriIndices[nodes[nodeIdx].leftOrFirstTri + i]];
		glm::vec3 centroid = (positions[t.vertIndices[0]].position + positions[t.vertIndices[1]].position + positions[t.vertIndices[2]].position) * 0.3333f;

		if (centroid[axis] < pos)
		{
			leftCount++;
			leftBox.grow(positions[t.vertIndices[0]].position);
			leftBox.grow(positions[t.vertIndices[1]].position);
			leftBox.grow(positions[t.vertIndices[2]].position);
		}
		else
		{
			rightCount++;
			rightBox.grow(positions[t.vertIndices[0]].position);
			rightBox.grow(positions[t.vertIndices[1]].position);
			rightBox.grow(positions[t.vertIndices[2]].position);
		}
	}
	float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
	return cost > 0 ? cost : 1e30f;
}


float BVH::CalculateNodeCost(int nodeIdx)
{
	glm::vec3 e = nodes[nodeIdx].boxMax - nodes[nodeIdx].boxMin; // extent of the node
	float surfaceArea = 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
	return nodes[nodeIdx].primCount * surfaceArea;
}


void BVH::Subdivide(int nodeIdx) {
	glm::vec3 diag = (nodes[nodeIdx].boxMax - nodes[nodeIdx].boxMin);


	int axis = -1;
	float split = -1;
	float cost = -1;
	FindBestSplit(nodeIdx, axis, split, cost);

	int i = nodes[nodeIdx].leftOrFirstTri;
	int j = i + nodes[nodeIdx].primCount - 1;
	while (i <= j) {
		int triIndex = sortedTriIndices[i];
		Triangle t = triangles[triIndex];
		glm::vec3 centroid = (positions[t.vertIndices[0]].position + positions[t.vertIndices[1]].position + positions[t.vertIndices[2]].position) * 0.3333f;
		if (centroid[axis] <= split) {
			i++;
		}
		else {
			std::swap(sortedTriIndices[i], sortedTriIndices[j]);
			j--;
		}
	}

	int leftCount = i - nodes[nodeIdx].leftOrFirstTri;
	int leftStart = nodes[nodeIdx].leftOrFirstTri;

	//no division occured
	if (leftCount == 0 || leftCount == nodes[nodeIdx].primCount) return;

	int rightStart = i;
	int rightCount = nodes[nodeIdx].primCount - leftCount;

	glm::vec3 e = nodes[nodeIdx].boxMax - nodes[nodeIdx].boxMin; // extent of parent
	float parentCost = CalculateNodeCost(nodeIdx);
	if (cost >= parentCost) return;

	nodes[nodeIdx].leftOrFirstTri = nodes.size();
	nodes[nodeIdx].primCount = 0;

	nodes.push_back(BVHNode());
	nodes.push_back(BVHNode());

	nodes[nodes[nodeIdx].leftOrFirstTri].leftOrFirstTri = leftStart;
	nodes[nodes[nodeIdx].leftOrFirstTri].primCount = leftCount;

	nodes[nodes[nodeIdx].leftOrFirstTri + 1].leftOrFirstTri = rightStart;
	nodes[nodes[nodeIdx].leftOrFirstTri + 1].primCount = rightCount;

	UpdateBounds(nodes[nodeIdx].leftOrFirstTri);
	UpdateBounds(nodes[nodeIdx].leftOrFirstTri + 1);

	if (nodes[nodes[nodeIdx].leftOrFirstTri].primCount > 1) {
		Subdivide(nodes[nodeIdx].leftOrFirstTri);
	}

	if (nodes[nodes[nodeIdx].leftOrFirstTri + 1].primCount > 1) {
		Subdivide(nodes[nodeIdx].leftOrFirstTri + 1);
	}
}

void BVH::FindBestSplit(int nodeIdx, int& chosenAxis, float& chosenPosition, float& chosenCost)
{
	int bestAxis = -1;
	float bestPosition = 0;
	float bestCost = FLT_MAX;

	for (int axis = 0; axis < 3; ++axis) {
		float min = nodes[nodeIdx].boxMin[axis];
		float max = nodes[nodeIdx].boxMax[axis];

		if (min == max) continue;
		float scale = (max - min) / 100.f;
		for (int i = 1; i < 100; ++i) {
			float candidatePosition = min + i * scale;
			float cost = EvaluateSAH(nodeIdx, axis, candidatePosition);
			if (cost < bestCost) {
				bestCost = cost;
				bestAxis = axis;
				bestPosition = candidatePosition;
			}
		}

		//for (int triIndex = 0; triIndex < nodes[nodeIdx].primCount; ++triIndex) {
		//	Triangle t = triangles[sortedTriIndices[nodes[nodeIdx].leftOrFirstTri + triIndex]];
		//	glm::vec3 centroid = (positions[t.vertIndices[0]].position + positions[t.vertIndices[1]].position + positions[t.vertIndices[2]].position) * 0.3333f;
		//	float candidatePosition = centroid[axis];
		//	float cost = EvaluateSAH(nodes[nodeIdx], axis, candidatePosition);
		//	if (cost < bestCost) {
		//		bestPosition = candidatePosition;
		//		bestAxis = axis;
		//		bestCost = cost;
		//	}
		//}
	}

	chosenAxis = bestAxis;
	chosenPosition = bestPosition;
	chosenCost = bestCost;
}

