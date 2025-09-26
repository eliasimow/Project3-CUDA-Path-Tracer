#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

//following:
//https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
__host__ __device__ float triangleIntersectionTest(
    Ray r,
    int triangleIdx,
    const Triangle* __restrict__ triangles,
    const glm::vec3* __restrict__ positions,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Triangle tri = triangles[triangleIdx];
    glm::vec3 pos1 = positions[tri.vertIndices[0]];
    glm::vec3 pos2 = positions[tri.vertIndices[1]];
    glm::vec3 pos3 = positions[tri.vertIndices[2]];

    glm::vec3 edge1 = pos2 - pos1;
    glm::vec3 edge2 = pos3 - pos1;

    glm::vec3 pVec = glm::cross(r.direction, edge2);
    float determinate = glm::dot(edge1, pVec);

    if (determinate > -FLT_EPSILON && determinate < FLT_EPSILON) {
        return -1;
    }

    float inverseDeterminate = 1.0 / determinate;
    glm::vec3 tVec = r.origin - pos1;

    float u = glm::dot(tVec, pVec) * inverseDeterminate;
    if (u < 0 || u > 1.) {
        //opposite direction, outta here
        return -1;
    }

    glm::vec3 qVec = glm::cross(tVec, edge1);

    float v = glm::dot(r.direction, qVec) * inverseDeterminate;
    if (v < 0 || u + v > 1) {
        //behind or past;
        return -1;
    }

    float t = glm::dot(edge2, qVec) * inverseDeterminate;
    intersectionPoint = r.origin + r.direction * t;

    //todo: barycentric interpolation. should this happen in material determination? 
    normal = glm::cross(edge1, edge2);
    outside = glm::dot(normal, r.direction) < FLT_EPSILON;

    return t;
}


__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = glm::min(t1, t2);
        outside = true;
    }
    else
    {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
    //    normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool intersectAABB(const Ray ray, float t, const glm::vec3 bmin, const glm::vec3 bmax)
{
    float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
    float tmin = glm::min(tx1, tx2), tmax = glm::max(tx1, tx2);
    float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
    tmin = glm::max(tmin, glm::min(ty1, ty2)), tmax = glm::min(tmax, glm::max(ty1, ty2));
    float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
    tmin = glm::max(tmin, glm::min(tz1, tz2)), tmax = glm::min(tmax, glm::max(tz1, tz2));
    return tmax >= tmin && (tmin < t || t < 0) && tmax > 0;
}

__host__ __device__ float intersectBVH(
    Ray ray, 
    const BVHNode* __restrict__  nodes,
    const Triangle* __restrict__ triangles,
    const glm::vec3* __restrict__ positions,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside) {

    float tTest = -1;
    float t = -1;
    glm::vec3 intersectionTest;
    glm::vec3 normalTest;
    bool outsideTest;

    int stack[32];
    int stackIndex = 1;
    stack[0] = 0;

    while (stackIndex > 0) {
        stackIndex--;
        BVHNode node = nodes[stack[stackIndex]];
        
        bool boxIntersect = intersectAABB(ray, t, node.boxMin, node.boxMax);

        if (!boxIntersect) continue;

        if (node.primCount > 0)
        {
            for (int i = 0; i < node.primCount; i++) {
                tTest = triangleIntersectionTest(ray, node.firstIndex + i, triangles, positions, intersectionTest, normalTest, outsideTest);

                if (tTest > 0 && !boxIntersect) {
                    return -10;
                }

                if (tTest > 0 && (tTest < t || t < 0)) {
                    t = tTest;
                    intersectionPoint = intersectionTest;
                    normal = normalTest;
                    outside = outsideTest;
                }
            }
        }else{
            stack[stackIndex++] = node.left;
            stack[stackIndex++] = node.right;
        }
    }
    return t;
}


