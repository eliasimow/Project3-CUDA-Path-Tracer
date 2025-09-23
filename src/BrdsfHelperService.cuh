#pragma once
#include <glm/glm.hpp>
#include <algorithm>
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <crt/host_defines.h>

__device__ const float M_PI = 3.14159265359;
__device__ const float I_PI = 0.31830988618f;

__host__ __device__ inline const float Cos2Theta(const glm::vec3 w) { return glm::pow(w.z, 2); }
__host__ __device__ inline const float Sin2Theta(const glm::vec3 w) { return std::max<float>(0, 1 - Cos2Theta(w)); }
__host__ __device__ inline const float SinTheta(const glm::vec3 w) { return std::sqrt(Sin2Theta(w)); }
__host__ __device__ inline const float CosTheta(const glm::vec3 w) { return w.z; }
__host__ __device__ inline const float AbsCosTheta(const glm::vec3 w) { return std::abs(w.z); }
__host__ __device__ inline const float tanTheta(const glm::vec3 w) { return SinTheta(w) / CosTheta(w); }
__host__ __device__ inline const float tan2Theta(const glm::vec3 w) { return Sin2Theta(w) / Cos2Theta(w); }
__host__ __device__ inline const float SafeACos(const float x) { return std::acos(glm::clamp(x, -1.f, 1.f)); }

__host__ __device__ inline const float CosPhi(const glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}

__host__ __device__ inline const float SinPhi(const glm::vec3 w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}

__host__ __device__ inline const glm::vec3 SphericalDirection(const float sinTheta, const float cosTheta, const float phi) {
    return glm::vec3(
        glm::clamp(sinTheta, -1.f, 1.f) * std::cos(phi),
        glm::clamp(sinTheta, -1.f, 1.f) * std::sin(phi),
        glm::clamp(cosTheta, -1.f, 1.f)
    );
}

__host__ __device__ inline const float CosDPhi(const glm::vec3 wa, const glm::vec3 wb) {
    float waxy = glm::pow(wa.x, 2) + glm::pow(wa.y, 2);
    float wbxy = glm::pow(wb.x, 2) + glm::pow(wb.y, 2);
    if (waxy == 0 || wbxy == 0) return 1;
    return glm::clamp((wa.x * wb.x + wa.y * wb.y) / std::sqrt(waxy * wbxy), -1.f, 1.f);
}

__host__ __device__ inline const float CosineHemispherePDF(const float cosTheta) {
    return cosTheta * I_PI;
}

__host__ __device__ inline const glm::vec3 Reflect(const glm::vec3 wo, const glm::vec3 n) {
    return -wo + 2 * glm::dot(wo, n) * n;
}

__host__ __device__ inline const float SphericalTheta(const glm::vec3 v) { return SafeACos(v.z); }

__host__ __device__ inline const float SphericalPhi(const glm::vec3 v) {
    float p = std::atan2(v.y, v.x);
    return (p < 0) ? (p + 2 * M_PI) : p;
}

__host__ __device__ inline glm::vec3 getLocalPath(const glm::vec3& path, const glm::vec3& intersectionNormal) {
    glm::vec3 normal = glm::normalize(intersectionNormal);
    glm::vec3 helper = (fabs(normal.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::vec3 tangent = glm::normalize(glm::cross(normal, helper));
    glm::vec3 bitangent = glm::normalize(glm::cross(tangent, normal)); // Note: tangent first

    // Transform using dot products (more explicit)
    return glm::vec3(
        glm::dot(path, tangent),
        glm::dot(path, bitangent),
        glm::dot(path, normal)
    );
}

__host__ __device__ inline glm::vec3 getGlobalPath(const glm::vec3& path, const glm::vec3& intersectionNormal) {
    glm::vec3 normal = glm::normalize(intersectionNormal);
    glm::vec3 helper = (fabs(normal.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::vec3 tangent = glm::normalize(glm::cross(normal, helper));
    glm::vec3 bitangent = glm::normalize(glm::cross(tangent, normal));

    return tangent * path.x + bitangent * path.y + normal * path.z;
}
