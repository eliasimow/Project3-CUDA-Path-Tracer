#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <tiny_gltf.h>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.materialType = DIFFUSE;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.materialType = EMISSION;

        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.materialType = SPECULAR;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    Gltf parser;
    //"C:\Users\elias\Downloads\animated_dance_teacher_-_bellydance.zip"
    FullGltfData gltfData = parser.LoadFromFile("C:/Users/elias/Downloads/animated_dance_teacher_-_bellydance/sceneTest.gltf");
    BufferMesh(gltfData.meshes);
    BuildBVH();

    Geom meshGeom;
    meshGeom.type = TRIANGLES;
    meshGeom.materialid = 1;
    geoms.push_back(meshGeom);


    currentFrame = 0;
    totalFrames = fps * gltfData.animationTime + 1.;


    //find environment map
    if (data.contains("EnvironmentMap")) {
        const auto& environmentData = data["EnvironmentMap"];
        std::string path = environmentData["PATH"];
        parseTextureFromPath(path, environmentWidth, environmentHeight, environmentTexture);
    }
}

void Scene::BuildBVH()
{
    bvh = std::make_unique<BVH>(triangles, vertPos);
    bvh->BuildBVH();

    std::vector<Triangle> reordered(triangles.size());
    for (size_t i = 0; i < triangles.size(); i++) {
        reordered[i] = triangles[bvh->sortedTriIndices[i]];
    }
    triangles = reordered;
}

void Scene::BufferMesh(std::vector<Mesh> meshes) {
    for each (Mesh m in meshes){
        
        //Material mat;
        //mat.materialType = DIFFUSE;
        //mat.color = glm::vec3(0.9, 0.1, 0.1);
        //materials.push_back(mat);

        //todo: more of these
        Geom meshGeometry;
        meshGeometry.transform = glm::mat4();
        meshGeometry.inverseTransform = glm::mat4();
        meshGeometry.materialid = materials.size() - 1;
        meshGeometry.type = MESH;

        int indexOffset = vertPos.size();

        for (int i = 0; i < m.positions.size(); ++i) {
            vertPos.push_back(m.positions[i]);
        }

        for (int i = 0; i < m.indices.size() - 2; i += 3) {
            Triangle t;
            t.vertIndices[0] = indexOffset + m.indices[i];
            t.vertIndices[1] = indexOffset + m.indices[i+1];
            t.vertIndices[2] = indexOffset + m.indices[i+2];
            triangles.push_back(t);
        }
    }
}

void Scene::IterateFrame()
{
    currentFrame++;
}
