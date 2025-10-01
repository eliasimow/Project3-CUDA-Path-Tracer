#pragma once
#include "scene.h"

class AnimationParser {
public:

	static void UpdateLocalMatrices(Scene& scene, float currentTime);

	/// <summary>
	/// Update joint global matrices according to their local matrices and hierarchy
	/// </summary>
	/// <param name="scene"></param>
	static void UpdateGlobalMatrices(Scene& scene, int nodeIndex);

	static void UpdateVerticesAndNormals(Scene& scene, Mesh& mesh);
};