#include "AnimationParser.h"
#include <glm/gtx/transform.hpp>


static glm::vec4 CalculateInterpolationValue(InterpolationType type, glm::vec4 pre, glm::vec4 post,
	float startTime,
	float currentTime,
	float endTime,
	const glm::vec4& inTangent0 = glm::vec4(),
	const glm::vec4& outTangent0 = glm::vec4(),
	const glm::vec4& inTangent1 = glm::vec4(),
	const glm::vec4& outTangent1 = glm::vec4()
	) {

	float dt = endTime - startTime;
	float t = (currentTime - startTime) / (dt);

	float t2 = t * t;
	float t3 = t2 * t;

	float h00 = 2 * t3 - 3 * t2 + 1;
	float h10 = t3 - 2 * t2 + t;
	float h01 = -2 * t3 + 3 * t2;
	float h11 = t3 - t2;

	glm::quat rot;

	switch (type) {
	case ROTATIONLINEAR:
		rot = glm::slerp(glm::quat(pre.w,pre.x,pre.y,pre.z), glm::quat(post.w, post.x, post.y, post.z), t);
		return glm::vec4(rot.x, rot.y, rot.z, rot.w);
		break;
	case LINEAR:
		return pre * (1 - t) + post * t;
		break;
	case STEP:
		return pre;
		break;
	case SPLINE:

		// dt = time difference between k0 and k1
		return h00 * pre +
			h10 * dt * outTangent0 +
			h01 * post +
			h11 * dt * inTangent1;

		break;
	default:
		break;
	}
}

void AnimationParser::UpdateLocalMatrices(Scene& scene, float currentTime) {
	FullGltfData& gltfData = scene.gltfData;

	for each (const Animation anim in gltfData.animations)
	{
		for each (const AnimationChannel channel in anim.channels) {
			//find current pre and post time:
			int preIndex = -1;
			int postIndex = -1;

			bool foundPost = false;
			for (int i = 0; i < channel.times.size() && !foundPost; ++i) {
				//yet to happen:
				if (channel.times[i] > currentTime) {
					postIndex = i;
					preIndex = i - 1;
					foundPost = true;
				}
			}


			//found a valid, mid interpolation channel
			if (preIndex >= 0) {
				float startTime = channel.times[preIndex];
				float endTime = channel.times[postIndex];

				//determine t:
				//can't be 0 divisor bc animation channels can't repeat times
				float t = (currentTime - startTime) / (endTime - startTime);
				glm::vec4 interpolated = channel.interpolationType != SPLINE ?
					CalculateInterpolationValue(channel.interpolationType, channel.values[preIndex], channel.values[postIndex], startTime, currentTime, endTime) :
					CalculateInterpolationValue(channel.interpolationType, channel.values[preIndex * 3 + 1], channel.values[postIndex * 3 + 1], startTime, currentTime, endTime,
						channel.values[preIndex * 3], channel.values[preIndex * 3 + 2],
						channel.values[postIndex * 3], channel.values[postIndex * 3 + 2]
					);

				Node& currentNode = gltfData.nodes[channel.targetNode];

				//okay, update the node current value now?
				switch (channel.path){
				case POSITION:
					currentNode.translation = glm::vec3(interpolated.x, interpolated.y, interpolated.z);
					break;
				case ROTATION:
					currentNode.rotation = glm::quat(interpolated.w, interpolated.x, interpolated.y, interpolated.z);
					break;
				case SCALE:
					currentNode.scale = glm::vec3(interpolated.x, interpolated.y, interpolated.z);
					break;
				default:
					break;			
				}
		
				currentNode.localMatrix =
					glm::translate(glm::mat4(1.0f), currentNode.translation) *
					glm::mat4_cast(currentNode.rotation) *
					glm::scale(glm::mat4(1.0f), currentNode.scale);
			}
		}
	}
}

void AnimationParser::UpdateGlobalMatrices(Scene &scene, int nodeIndex){
	Node& node = scene.gltfData.nodes[nodeIndex];
	if (node.parent == -1) {
		node.globalMatrix = node.localMatrix;
	}
	else {
		Node& parent = scene.gltfData.nodes[node.parent];
		node.globalMatrix = parent.globalMatrix * node.localMatrix;
	}

	for (int childIndex : node.children) {
		UpdateGlobalMatrices(scene, childIndex);
	}
}

void AnimationParser::UpdateVerticesAndNormals(Scene& scene, Mesh& mesh)
{
	for (int i = 0; i < mesh.positions.size() ; ++i) {
		glm::vec3 bindPos = mesh.bindVertPos[i];
		glm::vec3 bindNorm = mesh.bindNormals[i];
		glm::vec4 newPos = glm::vec4(0.0f);
		glm::vec4 newNorm = glm::vec4(0.0f);
	
		if (i >= mesh.jointIndices.size() || i >= mesh.weights.size()) {
			continue;
		}

		glm::ivec4 joints = mesh.jointIndices[i];
		glm::vec4 weights = mesh.weights[i];
		Skin skin = mesh.skin;
	
		for (int j = 0; j < 4; ++j) {
			int jointIndex = joints[j];
			float weight = weights[j];
			if (weight > 0.0f) {
				glm::mat4 jointMatrix = scene.gltfData.nodes[skin.joints[jointIndex]].globalMatrix;
				glm::mat4 skinMatrix = jointMatrix * skin.inverseBindMatrices[jointIndex];
				newPos += weight * (skinMatrix * glm::vec4(bindPos, 1.0f));
				glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(skinMatrix)));
				newNorm += weight * glm::vec4(normalMatrix * bindNorm, 0.0f);
			}
		}
	
		mesh.positions[i] = glm::vec3(newPos.x, newPos.y, newPos.z);
		mesh.normals[i] = glm::normalize(glm::vec3(newNorm.x, newNorm.y, newNorm.z));
	}
}
