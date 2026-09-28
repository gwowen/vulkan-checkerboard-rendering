#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};

inline void addTriangle(std::vector<uint32_t>& indices, uint32_t a, uint32_t b, uint32_t c)
{
    indices.insert(indices.end(), {a, b, c});
}

inline void addQuad(std::vector<uint32_t>& indices, uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    addTriangle(indices, a, b, c);
    addTriangle(indices, a, c, d);
}

inline void addBox(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, const glm::mat4& xform,
                   const glm::vec3& halfExtents, const glm::vec3& color)
{
    const glm::mat3 normalMat(xform);
    struct Face
    {
        glm::vec3 normal;
        glm::vec3 origin;
        glm::vec3 u;
        glm::vec3 v;
    };
    const Face faces[] = {
        {{1, 0, 0}, {halfExtents.x, 0, 0}, {0, 0, -halfExtents.z}, {0, halfExtents.y, 0}},
        {{-1, 0, 0}, {-halfExtents.x, 0, 0}, {0, 0, halfExtents.z}, {0, halfExtents.y, 0}},
        {{0, 1, 0}, {0, halfExtents.y, 0}, {halfExtents.x, 0, 0}, {0, 0, -halfExtents.z}},
        {{0, -1, 0}, {0, -halfExtents.y, 0}, {halfExtents.x, 0, 0}, {0, 0, halfExtents.z}},
        {{0, 0, 1}, {0, 0, halfExtents.z}, {halfExtents.x, 0, 0}, {0, halfExtents.y, 0}},
        {{0, 0, -1}, {0, 0, -halfExtents.z}, {-halfExtents.x, 0, 0}, {0, halfExtents.y, 0}},
    };

    for (const Face& face : faces)
    {
        const uint32_t base = static_cast<uint32_t>(vertices.size());
        const glm::vec3 corners[4] = {
            face.origin - face.u - face.v,
            face.origin + face.u - face.v,
            face.origin + face.u + face.v,
            face.origin - face.u + face.v,
        };
        const glm::vec3 worldNormal = glm::normalize(normalMat * face.normal);
        for (const glm::vec3& local : corners)
        {
            vertices.push_back({glm::vec3(xform * glm::vec4(local, 1.0f)), worldNormal, color});
        }
        addQuad(indices, base, base + 1, base + 2, base + 3);
    }
}

inline void addDisc(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, const glm::vec3& center,
                    const glm::vec3& normal, float radius, int segments, const glm::vec3& color, float scaleX = 1.0f,
                    float scaleZ = 1.0f)
{
    const glm::vec3 n = glm::normalize(normal);
    glm::vec3 tangent = std::fabs(n.y) < 0.99f ? glm::normalize(glm::cross(n, glm::vec3(0, 1, 0)))
                                               : glm::normalize(glm::cross(n, glm::vec3(1, 0, 0)));
    const glm::vec3 bitangent = glm::normalize(glm::cross(n, tangent));
    const uint32_t centerIndex = static_cast<uint32_t>(vertices.size());
    vertices.push_back({center, n, color});
    for (int i = 0; i < segments; ++i)
    {
        const float angle = (static_cast<float>(i) / static_cast<float>(segments)) * 6.2831853f;
        const glm::vec3 pos = center + (tangent * std::cos(angle) * radius * scaleX) +
                              (bitangent * std::sin(angle) * radius * scaleZ);
        vertices.push_back({pos, n, color});
    }
    for (int i = 0; i < segments; ++i)
    {
        const uint32_t a = centerIndex + 1 + static_cast<uint32_t>(i);
        const uint32_t b = centerIndex + 1 + static_cast<uint32_t>((i + 1) % segments);
        addTriangle(indices, centerIndex, a, b);
    }
}

inline void addCylinder(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, const glm::vec3& a,
                        const glm::vec3& b, float radius, int segments, const glm::vec3& color)
{
    const glm::vec3 axis = b - a;
    const float height = glm::length(axis);
    if (height < 1e-5f)
    {
        return;
    }
    const glm::vec3 n = axis / height;
    glm::vec3 tangent = std::fabs(n.y) < 0.99f ? glm::normalize(glm::cross(n, glm::vec3(0, 1, 0)))
                                               : glm::normalize(glm::cross(n, glm::vec3(1, 0, 0)));
    const glm::vec3 bitangent = glm::normalize(glm::cross(n, tangent));
    const uint32_t base = static_cast<uint32_t>(vertices.size());
    for (int i = 0; i < segments; ++i)
    {
        const float angle = (static_cast<float>(i) / static_cast<float>(segments)) * 6.2831853f;
        const glm::vec3 offset = (tangent * std::cos(angle) + bitangent * std::sin(angle)) * radius;
        const glm::vec3 normal = glm::normalize(offset);
        vertices.push_back({a + offset, normal, color});
        vertices.push_back({b + offset, normal, color});
    }
    for (int i = 0; i < segments; ++i)
    {
        const uint32_t i0 = base + static_cast<uint32_t>(i * 2);
        const uint32_t i1 = base + static_cast<uint32_t>(((i + 1) % segments) * 2);
        addQuad(indices, i0, i1, i1 + 1, i0 + 1);
    }
    addDisc(vertices, indices, a, -n, radius, segments, color);
    addDisc(vertices, indices, b, n, radius, segments, color);
}

inline void addLathe(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                     const std::vector<glm::vec2>& profile, int segments, const glm::vec3& color,
                     const glm::vec3& offset = glm::vec3(0.0f), float scaleX = 1.0f, float scaleZ = 1.0f,
                     float angleStart = 0.0f, float angleEnd = 6.2831853f, bool wrap = true)
{
    if (profile.size() < 2 || segments < 3)
    {
        return;
    }

    const int rings = wrap ? segments : segments + 1;
    const uint32_t base = static_cast<uint32_t>(vertices.size());
    const float angleSpan = angleEnd - angleStart;

    for (size_t i = 0; i < profile.size(); ++i)
    {
        const glm::vec2 prev = profile[i > 0 ? i - 1 : i];
        const glm::vec2 next = profile[i + 1 < profile.size() ? i + 1 : i];
        const glm::vec2 deriv = next - prev;
        for (int s = 0; s < rings; ++s)
        {
            const float t = wrap ? static_cast<float>(s) / static_cast<float>(segments)
                                 : static_cast<float>(s) / static_cast<float>(segments);
            const float angle = angleStart + angleSpan * t;
            const float c = std::cos(angle);
            const float sn = std::sin(angle);
            const glm::vec3 position = offset + glm::vec3(profile[i].x * c * scaleX, profile[i].y, profile[i].x * sn * scaleZ);
            const glm::vec3 tangent(deriv.x * c * scaleX, deriv.y, deriv.x * sn * scaleZ);
            const glm::vec3 hoop(-sn * scaleX, 0.0f, c * scaleZ);
            glm::vec3 normal = glm::cross(tangent, hoop);
            if (glm::dot(normal, normal) < 1e-8f)
            {
                normal = glm::vec3(c * scaleX, 0.0f, sn * scaleZ);
            }
            vertices.push_back({position, glm::normalize(normal), color});
        }
    }

    const uint32_t stride = static_cast<uint32_t>(rings);
    for (size_t i = 0; i + 1 < profile.size(); ++i)
    {
        for (int s = 0; s < segments; ++s)
        {
            const uint32_t s0 = static_cast<uint32_t>(s);
            const uint32_t s1 = wrap ? static_cast<uint32_t>((s + 1) % segments) : s0 + 1;
            const uint32_t a = base + static_cast<uint32_t>(i) * stride + s0;
            const uint32_t b = base + static_cast<uint32_t>(i) * stride + s1;
            const uint32_t c = base + static_cast<uint32_t>(i + 1) * stride + s1;
            const uint32_t d = base + static_cast<uint32_t>(i + 1) * stride + s0;
            addQuad(indices, a, b, c, d);
        }
    }

    if (!wrap)
    {
        for (int end = 0; end < 2; ++end)
        {
            const int s = end == 0 ? 0 : segments;
            const float angle = angleStart + angleSpan * (end == 0 ? 0.0f : 1.0f);
            const glm::vec3 capNormal = glm::normalize(glm::vec3(-std::sin(angle) * scaleX, 0.0f, std::cos(angle) * scaleZ)) *
                                        (end == 0 ? -1.0f : 1.0f);
            const uint32_t capBase = static_cast<uint32_t>(vertices.size());
            for (size_t i = 0; i < profile.size(); ++i)
            {
                Vertex vertex = vertices[base + static_cast<uint32_t>(i) * stride + static_cast<uint32_t>(s)];
                vertex.normal = capNormal;
                vertices.push_back(vertex);
            }
            for (size_t i = 0; i + 2 < profile.size(); ++i)
            {
                if (end == 0)
                {
                    addTriangle(indices, capBase, capBase + static_cast<uint32_t>(i + 2), capBase + static_cast<uint32_t>(i + 1));
                }
                else
                {
                    addTriangle(indices, capBase, capBase + static_cast<uint32_t>(i + 1), capBase + static_cast<uint32_t>(i + 2));
                }
            }
        }
    }
}

inline void centerAndScaleMesh(std::vector<Vertex>& vertices, float targetHeight)
{
    glm::vec3 minP(1e9f);
    glm::vec3 maxP(-1e9f);
    for (const Vertex& vertex : vertices)
    {
        minP = glm::min(minP, vertex.position);
        maxP = glm::max(maxP, vertex.position);
    }
    const glm::vec3 center = (minP + maxP) * 0.5f;
    const float height = std::max(maxP.y - minP.y, 1e-4f);
    const float scale = targetHeight / height;
    for (Vertex& vertex : vertices)
    {
        vertex.position = (vertex.position - center) * scale;
    }
}

inline void addColoredCubeFace(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, const glm::vec3& normal,
                               const glm::vec3& color, const glm::vec3& origin, const glm::vec3& u, const glm::vec3& v)
{
    const uint32_t base = static_cast<uint32_t>(vertices.size());
    vertices.push_back({origin - u - v, normal, color});
    vertices.push_back({origin + u - v, normal, color});
    vertices.push_back({origin + u + v, normal, color});
    vertices.push_back({origin - u + v, normal, color});
    addQuad(indices, base, base + 1, base + 2, base + 3);
}

inline void createCubeMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
{
    vertices.clear();
    indices.clear();
    const float h = 0.5f;
    addColoredCubeFace(vertices, indices, {1, 0, 0}, {0.91f, 0.34f, 0.34f}, {h, 0, 0}, {0, 0, -h}, {0, h, 0});
    addColoredCubeFace(vertices, indices, {-1, 0, 0}, {0.33f, 0.76f, 0.45f}, {-h, 0, 0}, {0, 0, h}, {0, h, 0});
    addColoredCubeFace(vertices, indices, {0, 1, 0}, {0.38f, 0.56f, 0.93f}, {0, h, 0}, {h, 0, 0}, {0, 0, -h});
    addColoredCubeFace(vertices, indices, {0, -1, 0}, {0.95f, 0.82f, 0.31f}, {0, -h, 0}, {h, 0, 0}, {0, 0, h});
    addColoredCubeFace(vertices, indices, {0, 0, 1}, {0.86f, 0.45f, 0.84f}, {0, 0, h}, {h, 0, 0}, {0, h, 0});
    addColoredCubeFace(vertices, indices, {0, 0, -1}, {0.35f, 0.84f, 0.86f}, {0, 0, -h}, {-h, 0, 0}, {0, h, 0});
}

inline void createToiletMesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
{
    vertices.clear();
    indices.clear();

    const glm::vec3 porcelain(0.93f, 0.93f, 0.90f);
    const glm::vec3 porcelainWarm(0.88f, 0.88f, 0.84f);
    const glm::vec3 seat(0.18f, 0.19f, 0.22f);
    const glm::vec3 water(0.27f, 0.58f, 0.70f);
    const glm::vec3 chrome(0.74f, 0.77f, 0.82f);

    const std::vector<glm::vec2> bowl = {
        {0.00f, 0.00f}, {0.13f, 0.00f}, {0.16f, 0.015f}, {0.155f, 0.06f}, {0.15f, 0.12f},
        {0.16f, 0.18f}, {0.21f, 0.22f}, {0.30f, 0.26f},  {0.37f, 0.31f},  {0.40f, 0.37f},
        {0.405f, 0.43f}, {0.41f, 0.46f}, {0.40f, 0.475f}, {0.36f, 0.48f}, {0.325f, 0.47f},
        {0.31f, 0.44f}, {0.30f, 0.36f}, {0.26f, 0.30f}, {0.21f, 0.25f}, {0.17f, 0.232f},
    };
    addLathe(vertices, indices, bowl, 48, porcelain, glm::vec3(0.0f, 0.0f, 0.04f), 1.05f, 1.18f);

    addDisc(vertices, indices, glm::vec3(0.0f, 0.234f, 0.04f), glm::vec3(0.0f, 1.0f, 0.0f), 0.17f, 32, water, 1.05f,
            1.18f);
    addDisc(vertices, indices, glm::vec3(0.0f, 0.233f, 0.04f), glm::vec3(0.0f, -1.0f, 0.0f), 0.17f, 32, water, 1.05f,
            1.18f);

    const std::vector<glm::vec2> seatProfile = {
        {0.255f, 0.478f}, {0.275f, 0.476f}, {0.292f, 0.482f}, {0.298f, 0.494f},
        {0.292f, 0.506f}, {0.275f, 0.512f}, {0.255f, 0.508f}, {0.248f, 0.494f}, {0.255f, 0.478f},
    };
    const float seatStart = -1.17f;
    const float seatEnd = 4.31f;
    addLathe(vertices, indices, seatProfile, 40, seat, glm::vec3(0.0f, 0.0f, 0.05f), 1.08f, 1.20f, seatStart, seatEnd,
             false);

    glm::mat4 lid = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.52f, -0.16f));
    lid = glm::rotate(lid, -0.62f, glm::vec3(1.0f, 0.0f, 0.0f));
    lid = glm::translate(lid, glm::vec3(0.0f, 0.015f, 0.17f));
    addBox(vertices, indices, lid, glm::vec3(0.20f, 0.012f, 0.22f), seat);

    addBox(vertices, indices, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.66f, -0.38f)),
           glm::vec3(0.205f, 0.20f, 0.085f), porcelain);
    addBox(vertices, indices, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.875f, -0.38f)),
           glm::vec3(0.215f, 0.018f, 0.095f), porcelainWarm);
    addBox(vertices, indices, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.50f, -0.26f)),
           glm::vec3(0.11f, 0.07f, 0.08f), porcelain);

    const glm::vec3 handleRoot(0.195f, 0.74f, -0.30f);
    addCylinder(vertices, indices, handleRoot, handleRoot + glm::vec3(0.07f, 0.0f, 0.0f), 0.012f, 12, chrome);
    addBox(vertices, indices, glm::translate(glm::mat4(1.0f), handleRoot + glm::vec3(0.09f, 0.0f, 0.02f)),
           glm::vec3(0.028f, 0.008f, 0.045f), chrome);

    addCylinder(vertices, indices, glm::vec3(-0.07f, 0.49f, -0.14f), glm::vec3(-0.07f, 0.52f, -0.14f), 0.012f, 10,
                chrome);
    addCylinder(vertices, indices, glm::vec3(0.07f, 0.49f, -0.14f), glm::vec3(0.07f, 0.52f, -0.14f), 0.012f, 10, chrome);

    centerAndScaleMesh(vertices, 1.35f);
}
