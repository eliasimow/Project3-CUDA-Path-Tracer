#include "GltfParse.h"
#include <iostream>
#include <tiny_gltf.h>
#include <vector_types.h>
#include <vector_functions.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"
#include <glm/gtx/transform.hpp>

namespace {

    inline bool IsFloatComponentType(int componentType) {
        return componentType == TINYGLTF_COMPONENT_TYPE_FLOAT;
    }

    const unsigned char* GetAccessorDataPointer(const tinygltf::Model& model, const tinygltf::Accessor& accessor, size_t& outByteStride) {
        if (accessor.bufferView < 0) return nullptr;
        const tinygltf::BufferView& bv = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buf = model.buffers[bv.buffer];
        size_t bufferViewOffset = bv.byteOffset;
        size_t accessorByteOffset = accessor.byteOffset;
        outByteStride = bv.byteStride ? static_cast<size_t>(bv.byteStride) : 0;
        return buf.data.data() + bufferViewOffset + accessorByteOffset;
    }

    // Read indices generically and return uint32_t vector (handles UBYTE/USHORT/UINT)
    std::vector<uint32_t> ReadIndices(const tinygltf::Model& model, const tinygltf::Primitive& prim) {
        if (prim.indices < 0) return {};
        const tinygltf::Accessor& acc = model.accessors[prim.indices];
        size_t byteStride = 0;
        const unsigned char* data = GetAccessorDataPointer(model, acc, byteStride);
        if (!data) return {};
        size_t count = acc.count;
        std::vector<uint32_t> out;
        out.reserve(count);

        switch (acc.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            for (size_t i = 0; i < count; ++i) {
                uint8_t v = *(reinterpret_cast<const uint8_t*>(data + i * (byteStride ? byteStride : sizeof(uint8_t))));
                out.push_back(static_cast<uint32_t>(v));
            }
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            for (size_t i = 0; i < count; ++i) {
                const uint16_t* ptr = reinterpret_cast<const uint16_t*>(data + i * (byteStride ? byteStride : sizeof(uint16_t)));
                out.push_back(static_cast<uint32_t>(*ptr));
            }
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            for (size_t i = 0; i < count; ++i) {
                const uint32_t* ptr = reinterpret_cast<const uint32_t*>(data + i * (byteStride ? byteStride : sizeof(uint32_t)));
                out.push_back(*ptr);
            }
            break;
        default:
            std::cerr << "Unsupported index component type: " << acc.componentType << "\n";
        }
        return out;
    }

    template <typename T>
    std::vector<T> ReadAccessor(const tinygltf::Model& model, int accessorIndex) {
        std::vector<T> result;
        if (accessorIndex < 0) return result;

        const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
        const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[view.buffer];

        const unsigned char* dataPtr = buffer.data.data() + accessor.byteOffset + view.byteOffset;
        size_t stride = accessor.ByteStride(view);
        size_t count = accessor.count;

        result.resize(count);
        for (size_t i = 0; i < count; i++) {
            memcpy(&result[i], dataPtr + stride * i, sizeof(T));
        }
        return result;
    }
}

FullGltfData Gltf::LoadFromFile(const std::string& path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ret;
    const std::string ext = path.size() >= 4 ? path.substr(path.size() - 4) : std::string();
    if (ext == ".glb") {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    }
    else {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }
    if (!warn.empty()) {
        std::cerr << "tinygltf warning: " << warn << "\n";
    }
    if (!err.empty()) {
        std::cerr << "tinygltf error: " << err << "\n";
    }
    if (!ret) {
        std::cerr << "Failed to load glTF: " << path << "\n";
        return FullGltfData();
    }

    //skins:
    std::vector<Skin> skins;
    for (auto& glSkin : model.skins) {
        Skin s;
        s.joints = glSkin.joints;
        s.skeletonRoot = glSkin.skeleton;

        if (glSkin.inverseBindMatrices >= 0) {
            auto mats = ReadAccessor<glm::mat4>(model, glSkin.inverseBindMatrices);
            s.inverseBindMatrices = mats;
        }
        skins.push_back(s);
    }

    std::unordered_map<int, int> meshIdToSkinId;

    //nodes:
    std::vector<Node> nodes(model.nodes.size());
    for (int i = 0; i < model.nodes.size(); ++i) {
        const auto& glNode = model.nodes[i];

        if (glNode.skin >= 0 && glNode.mesh >= 0) {
            meshIdToSkinId[glNode.mesh] = glNode.skin;
        }

        Node n;
        n.translation = glNode.translation.size() == 3 ?
            glm::vec3(glNode.translation[0], glNode.translation[1], glNode.translation[2]) :
            glm::vec3(0.0f);

        n.rotation = glNode.rotation.size() == 4 ?
            glm::quat(glNode.rotation[3], glNode.rotation[0], glNode.rotation[1], glNode.rotation[2]) :
            glm::quat(1, 0, 0, 0);

        n.scale = glNode.scale.size() == 3 ?
            glm::vec3(glNode.scale[0], glNode.scale[1], glNode.scale[2]) :
            glm::vec3(1.0f);

        n.children = glNode.children;

        // If node.matrix is set, override TRS
        glm::mat4 T = glm::translate(glm::mat4(1.0f), n.translation);
        glm::mat4 R = glm::mat4_cast(n.rotation);
        glm::mat4 S = glm::scale(glm::mat4(1.0f), n.scale);
        n.localMatrix = T * R * S;

        nodes[i] = n;
    }

    // set parents
    for (int i = 0; i < nodes.size(); i++) {
        for (int child : nodes[i].children) {
            nodes[child].parent = static_cast<int>(i);
        }
    }

    std::vector<Animation> animations;
    for (auto& glAnim : model.animations) {
        Animation anim;
        anim.name = glAnim.name;

        for (auto& ch : glAnim.channels) {
            const auto& sampler = glAnim.samplers[ch.sampler];

            AnimationChannel channel;
            channel.targetNode = ch.target_node;
            
            auto it = transformTypeMap.find(ch.target_path);
            channel.path = it != transformTypeMap.end() ? it->second : POSITION;

                
            auto interpolationSearch = interpolationTypeMap.find(sampler.interpolation);
            channel.interpolationType = interpolationSearch != interpolationTypeMap.end() ? interpolationSearch->second : LINEAR;

            if (channel.interpolationType == LINEAR && channel.path == ROTATION) {
                channel.interpolationType = ROTATIONLINEAR;
            }


            // Keyframe times
            channel.times = ReadAccessor<float>(model, sampler.input);

            // Values (vec3 for T/S, quat for R)
            if (ch.target_path == "rotation") {
                auto vals = ReadAccessor<glm::vec4>(model, sampler.output);
                channel.values = vals;
            }
            else {
                auto vals = ReadAccessor<glm::vec3>(model, sampler.output);
                channel.values.resize(vals.size());
                for (size_t i = 0; i < vals.size(); i++)
                    channel.values[i] = glm::vec4(vals[i], 0.0f);
            }

            anim.channels.push_back(channel);
        }
        animations.push_back(anim);
    }

    std::vector<Mesh> outMeshes;
    int vertexCount = 0;
    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            Mesh mymesh;
            mymesh.name = mesh.name;

            // Positions (required for rendering)
            if (prim.attributes.find("POSITION") != prim.attributes.end()) {
                const tinygltf::Accessor& acc = model.accessors[prim.attributes.at("POSITION")];
                std::vector<glm::vec3> pos = ReadAccessor<glm::vec3>(model, prim.attributes.at("POSITION"));
                size_t numElems = acc.count;
                mymesh.positions.resize(numElems);
                mymesh.bindVertPos.resize(numElems);

                for (size_t i = 0; i < numElems; ++i) {
                    mymesh.positions[i] = pos[i];
                    mymesh.bindVertPos[i] = pos[i];
                }
            }

            // Normals (optional)
            if (prim.attributes.find("NORMAL") != prim.attributes.end()) {
                const tinygltf::Accessor& acc = model.accessors[prim.attributes.at("NORMAL")];
                std::vector<glm::vec3> norms = ReadAccessor<glm::vec3>(model, prim.attributes.at("NORMAL"));
                size_t numElems = acc.count;
                mymesh.normals.resize(numElems);
                mymesh.bindNormals.resize(numElems);

                for (size_t i = 0; i < numElems; ++i) {
                    mymesh.normals[i] = norms[i];
                    mymesh.bindNormals[i] = norms[i];
                }
            }

            // Texcoords0
            if (prim.attributes.find("TEXCOORD_0") != prim.attributes.end()) {
                const tinygltf::Accessor& acc = model.accessors[prim.attributes.at("TEXCOORD_0")];
                std::vector<glm::vec2> uvs = ReadAccessor<glm::vec2>(model, prim.attributes.at("TEXCOORD_0"));
                size_t numElems = acc.count;
                mymesh.texcoords0.resize(numElems);
                for (size_t i = 0; i < numElems; ++i) {
                    mymesh.texcoords0[i] = uvs[i];
                }
            }

            //this feels really unsafe
            if (meshIdToSkinId.find(outMeshes.size()) != meshIdToSkinId.end()) {
                mymesh.skin = skins[meshIdToSkinId[outMeshes.size()]];
            }

            auto it = prim.attributes.find("JOINTS_0");
            if (it != prim.attributes.end()) {
                const tinygltf::Accessor& accessor = model.accessors[it->second];
                const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = model.buffers[view.buffer];

                size_t offset = accessor.byteOffset + view.byteOffset;
                const unsigned char* dataPtr = buffer.data.data() + offset;

                for (size_t i = 0; i < accessor.count; ++i) {
                    glm::uvec4 joint;
                    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(dataPtr + i * view.byteStride);
                        joint = glm::uvec4(ptr[0], ptr[1], ptr[2], ptr[3]);
                    }
                    else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(dataPtr + i * view.byteStride);
                        joint = glm::uvec4(ptr[0], ptr[1], ptr[2], ptr[3]);
                    }
                    else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(dataPtr + i * view.byteStride);
                        joint = glm::uvec4(ptr[0], ptr[1], ptr[2], ptr[3]);
                    }
                    mymesh.jointIndices.push_back(joint);
                }
            }

            auto weights = prim.attributes.find("WEIGHTS_0");
            if (weights != prim.attributes.end()) {
                mymesh.weights = ReadAccessor<glm::vec4>(model, weights->second);
            }

            // Indices
            mymesh.indices = ReadIndices(model, prim);

            // Material
            //if (prim.material >= 0 && prim.material < (int)materials.size()) {
            //    mymesh.material = materials[prim.material];
            //}

            mymesh.vertexOffset = vertexCount;
            vertexCount += mymesh.positions.size();

            outMeshes.push_back(std::move(mymesh));
        }
    }

    //when we have multiple meshes, we'll need to return to this and consider the index offsets. or maybe keep them separate. idk
    


    float animationTime = 0.f;
    if (animations.size() > 0) {
        for each (Animation anim in animations) {
            for each (AnimationChannel channel in anim.channels) {
                //good lord
                for each (float time in channel.times) {
                    animationTime = glm::max(animationTime, time);
                }
            }
        }
    }

    return FullGltfData(outMeshes, nodes, animations, animationTime);
}

void parseTextureFromPath(const std::string& path, int &width, int&height, std::vector<glm::vec4> &texture)
{
    int channels;

    float* imgData = stbi_loadf(path.c_str(), &width, &height, &channels, 0);

    if (!imgData) {
        //      printf("Failed to load image %s: %s\n", path, stbi_failure_reason());
    }
    else {
        texture.resize(width * height);
        for (int i = 0; i < width * height; ++i) {
            float r = imgData[3 * i + 0];
            float g = imgData[3 * i + 1];
            float b = imgData[3 * i + 2];
            texture[i] = glm::vec4(r, g, b, 1.0f);
        }
        stbi_image_free(imgData);
    }
}
