#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene* scene, bool ignoreGltfTextures = false);
void pathtraceFree(bool ignoreGltfTextues = false);
void pathtrace(uchar4* pbo, int frame, int iteration, SceneSettings settings);
void rewritePositions(Scene* scene);

