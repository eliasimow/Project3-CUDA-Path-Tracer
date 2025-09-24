#include "GltfParse.h"
#include <iostream>
#include <tiny_gltf.h>


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

    // Read floats (handles accessor's componentType float; basic support for normalized ubyte/ushort -> float)
    std::vector<float> ReadFloatAccessor(const tinygltf::Model& model, const tinygltf::Accessor& acc) {
        size_t byteStride = 0;
        const unsigned char* data = GetAccessorDataPointer(model, acc, byteStride);
        size_t compCount = tinygltf::GetNumComponentsInType(acc.type);
        if (!data) return {};
        std::vector<float> out;
        out.reserve(acc.count * compCount);

        size_t effectiveStride = byteStride ? byteStride : (compCount * (size_t)tinygltf::GetComponentSizeInBytes(acc.componentType));
        for (size_t i = 0; i < acc.count; ++i) {
            const unsigned char* elem = data + i * effectiveStride;
            switch (acc.componentType) {
            case TINYGLTF_COMPONENT_TYPE_FLOAT: {
                const float* fptr = reinterpret_cast<const float*>(elem);
                for (size_t c = 0; c < compCount; ++c) out.push_back(fptr[c]);
                break;
            }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                // normalized?
                for (size_t c = 0; c < compCount; ++c) {
                    uint8_t v = *(elem + c * sizeof(uint8_t));
                    float fv = acc.normalized ? (float(v) / 255.0f) : float(v);
                    out.push_back(fv);
                }
                break;
            }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                const uint16_t* uptr = reinterpret_cast<const uint16_t*>(elem);
                for (size_t c = 0; c < compCount; ++c) {
                    uint16_t v = uptr[c];
                    float fv = acc.normalized ? (float(v) / 65535.0f) : float(v);
                    out.push_back(fv);
                }
                break;
            }
            default:
                std::cerr << "Unsupported accessor component type for floats: " << acc.componentType << "\n";
                return {};
            }
        }
        return out;
    }

} // namespace

std::vector<Mesh> Gltf::LoadFromFile(const std::string& path){
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
        return {};
    }

    std::vector<Mesh> outMeshes;

    // load materials info (store for later use)
    //std::vector<MaterialInfo> materials;
    //materials.reserve(model.materials.size());
    //for (const auto& m : model.materials) {
    //    MaterialInfo mi;
    //    mi.name = m.name;
    //    if (m.values.find("baseColorTexture") != m.values.end()) {
    //        // tinygltf stores texture info differently depending on version, try to access pbrMetallicRoughness
    //    }
    //    // Better route: use pbrMetallicRoughness if present:
    //    if (m.values.find("baseColorFactor") != m.values.end()) {
    //        auto v = m.values.at("baseColorFactor").ColorFactor();
    //        mi.baseColorFactor = glm::vec4((float)v[0], (float)v[1], (float)v[2], (float)v[3]);
    //    }
    //    else if (m.pbrMetallicRoughness.baseColorFactor.size() == 4) {
    //        auto& v = m.pbrMetallicRoughness.baseColorFactor;
    //        mi.baseColorFactor = glm::vec4((float)v[0], (float)v[1], (float)v[2], (float)v[3]);
    //    }
    //    mi.metallicFactor = float(m.pbrMetallicRoughness.metallicFactor);
    //    mi.roughnessFactor = float(m.pbrMetallicRoughness.roughnessFactor);
    //    if (m.pbrMetallicRoughness.baseColorTexture.index >= 0) {
    //        int texIdx = m.pbrMetallicRoughness.baseColorTexture.index;
    //        if (texIdx >= 0 && texIdx < (int)model.textures.size()) {
    //            const tinygltf::Texture& t = model.textures[texIdx];
    //            if (t.source >= 0 && t.source < (int)model.images.size()) {
    //                mi.baseColorTexture = model.images[t.source].uri;
    //            }
    //        }
    //    }
    //    materials.push_back(mi);
    //}

    // Iterate nodes/meshes/primitives
    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            Mesh mymesh;
            mymesh.name = mesh.name;

            // Positions (required for rendering)
            if (prim.attributes.find("POSITION") != prim.attributes.end()) {
                const tinygltf::Accessor& acc = model.accessors[prim.attributes.at("POSITION")];
                auto floats = ReadFloatAccessor(model, acc);
                size_t numElems = acc.count;
                mymesh.positions.resize(numElems);
                for (size_t i = 0; i < numElems; ++i) {
                    float x = floats[i * 3 + 0];
                    float y = floats[i * 3 + 1];
                    float z = floats[i * 3 + 2];
                    mymesh.positions[i] = glm::vec3(x, y, z);
                }
            }

            // Normals (optional)
            if (prim.attributes.find("NORMAL") != prim.attributes.end()) {
                const tinygltf::Accessor& acc = model.accessors[prim.attributes.at("NORMAL")];
                auto floats = ReadFloatAccessor(model, acc);
                size_t numElems = acc.count;
                mymesh.normals.resize(numElems);
                for (size_t i = 0; i < numElems; ++i) {
                    mymesh.normals[i] = glm::vec3(floats[i * 3 + 0], floats[i * 3 + 1], floats[i * 3 + 2]);
                }
            }

            // Texcoords0
            if (prim.attributes.find("TEXCOORD_0") != prim.attributes.end()) {
                const tinygltf::Accessor& acc = model.accessors[prim.attributes.at("TEXCOORD_0")];
                auto floats = ReadFloatAccessor(model, acc);
                size_t numElems = acc.count;
                mymesh.texcoords0.resize(numElems);
                for (size_t i = 0; i < numElems; ++i) {
                    mymesh.texcoords0[i] = glm::vec2(floats[i * 2 + 0], floats[i * 2 + 1]);
                }
            }

            // Indices
            mymesh.indices = ReadIndices(model, prim);

            // Material
            //if (prim.material >= 0 && prim.material < (int)materials.size()) {
            //    mymesh.material = materials[prim.material];
            //}

            outMeshes.push_back(std::move(mymesh));
        }
    }

    return outMeshes;
}
