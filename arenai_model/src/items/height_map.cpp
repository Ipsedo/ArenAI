//
// Created by samuel on 23/03/2023.
//

#include "./height_map.h"

#include <algorithm>
#include <limits>

#include <glm/gtc/type_ptr.hpp>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>

#include "../jolt_engine.h"

using namespace arenai;
using namespace arenai::model;
using namespace arenai::utils;

namespace arenai::model {

    HeightMapItem::HeightMapItem(
        std::string name, JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &img_reader,
        const std::filesystem::path &height_map_file, glm::vec3 pos, glm::vec3 scale)
        : JoltItem(std::move(name), engine), shape_id(height_map_file.string()), scale(scale) {
        ImageChannels tmp = img_reader->read_png(height_map_file);
        auto [width, height, pixels] = utils::AbstractResourceFileReader::to_img_grey(tmp);

        map_width = width;
        map_height = height;

        image_grey = pixels;

        float min_height = std::numeric_limits<float>::infinity();
        float max_height = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < width * height; i++) {
            min_height = std::min(image_grey[i], min_height);
            max_height = std::max(image_grey[i], max_height);
        }

        build_render_mesh(
            glm::vec3(-2000.f, -2000.f, -2000.f), glm::vec3(2000.f, 2000.f, 2000.f), min_height,
            max_height);

        // collision mesh: same grid, same centering and diamond subdivision as
        // Bullet's btHeightfieldTerrainShape, triangles wound to face +Y (Jolt
        // meshes are single-sided)
        JPH::TriangleList triangles;
        triangles.reserve(2 * (map_height - 1) * (map_width - 1));

        auto to_jolt = [](const glm::vec3 &v) { return JPH::Float3(v.x, v.y, v.z); };

        auto push_triangle = [&](const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2) {
            const glm::vec3 normal = glm::cross(p1 - p0, p2 - p0);
            if (normal.y >= 0.f) triangles.emplace_back(to_jolt(p0), to_jolt(p1), to_jolt(p2));
            else triangles.emplace_back(to_jolt(p0), to_jolt(p2), to_jolt(p1));
        };

        for (int z = 0; z < map_height - 1; ++z) {
            for (int x = 0; x < map_width - 1; ++x) {
                const glm::vec3 p00 = make_pos(x, z, min_height, max_height);
                const glm::vec3 p10 = make_pos(x + 1, z, min_height, max_height);
                const glm::vec3 p01 = make_pos(x, z + 1, min_height, max_height);
                const glm::vec3 p11 = make_pos(x + 1, z + 1, min_height, max_height);

                // btHeightfieldTerrainShape with diamond subdivision alternates
                // the split diagonal on (x + z) parity
                if ((x + z) % 2 == 0) {
                    push_triangle(p00, p01, p11);
                    push_triangle(p00, p11, p10);
                } else {
                    push_triangle(p00, p01, p10);
                    push_triangle(p10, p01, p11);
                }
            }
        }

        const JPH::MeshShapeSettings mesh_settings(triangles);
        const JPH::ShapeRefC mesh_shape = mesh_settings.Create().Get();

        JPH::BodyCreationSettings body_settings(
            mesh_shape, JPH::RVec3(pos.x, pos.y, pos.z), JPH::Quat::sIdentity(),
            JPH::EMotionType::Static, layers::NON_MOVING);
        body_settings.mFriction = 0.5f;
        body_settings.mRestitution = 0.f;

        body = engine.get_body_interface().CreateBody(body_settings);
        body->SetUserData(reinterpret_cast<JPH::uint64>(static_cast<Item *>(this)));
    }

    float HeightMapItem::get_height(int x, int z) const {
        x = std::clamp(x, 0, map_width - 1);
        z = std::clamp(z, 0, map_height - 1);
        return image_grey[z * map_width + x];
    }

    glm::vec3 HeightMapItem::compute_vertex_normal(const int x, const int z) const {
        const float hL = get_height(x - 1, z);
        const float hR = get_height(x + 1, z);
        const float hD = get_height(x, z - 1);
        const float hU = get_height(x, z + 1);

        const glm::vec3 n(hL - hR, 2.0f * scale.x / scale.y, hD - hU);
        return glm::normalize(n);
    }

    static bool triangle_intersects_aabb(
        const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec3 &p3, const glm::vec3 &aabb_min,
        const glm::vec3 &aabb_max) {
        const float triMinX = std::min({p1.x, p2.x, p3.x});
        const float triMinY = std::min({p1.y, p2.y, p3.y});
        const float triMinZ = std::min({p1.z, p2.z, p3.z});

        const float triMaxX = std::max({p1.x, p2.x, p3.x});
        const float triMaxY = std::max({p1.y, p2.y, p3.y});
        const float triMaxZ = std::max({p1.z, p2.z, p3.z});

        if (triMaxX < aabb_min.x || triMinX > aabb_max.x) return false;
        if (triMaxY < aabb_min.y || triMinY > aabb_max.y) return false;
        if (triMaxZ < aabb_min.z || triMinZ > aabb_max.z) return false;

        return true;
    }

    glm::vec3 HeightMapItem::make_pos(
        const int x, const int z, const float min_height, const float max_height) const {
        const float half_width = static_cast<float>(map_width - 1) * 0.5f;
        const float half_length = static_cast<float>(map_height - 1) * 0.5f;
        const float center_y = 0.5f * (min_height + max_height);

        return {
            (static_cast<float>(x) - half_width) * scale.x, (get_height(x, z) - center_y) * scale.y,
            (static_cast<float>(z) - half_length) * scale.z};
    }

    void HeightMapItem::build_render_mesh(
        const glm::vec3 aabb_min, const glm::vec3 aabb_max, const float min_height,
        const float max_height) {
        vertices.clear();
        normals.clear();

        auto append_triangle = [this, &aabb_min, &aabb_max](
                                   const glm::vec3 &p1, const glm::vec3 &n1, const glm::vec3 &p2,
                                   const glm::vec3 &n2, const glm::vec3 &p3, const glm::vec3 &n3) {
            if (!triangle_intersects_aabb(p1, p2, p3, aabb_min, aabb_max)) return;

            vertices.emplace_back(p1.x, p1.y, p1.z);
            normals.emplace_back(n1.x, n1.y, n1.z);

            vertices.emplace_back(p2.x, p2.y, p2.z);
            normals.emplace_back(n2.x, n2.y, n2.z);

            vertices.emplace_back(p3.x, p3.y, p3.z);
            normals.emplace_back(n3.x, n3.y, n3.z);
        };

        for (int z = 0; z < map_height - 1; ++z) {
            for (int x = 0; x < map_width - 1; ++x) {
                const glm::vec3 p00 = make_pos(x, z, min_height, max_height);
                const glm::vec3 p10 = make_pos(x + 1, z, min_height, max_height);
                const glm::vec3 p01 = make_pos(x, z + 1, min_height, max_height);
                const glm::vec3 p11 = make_pos(x + 1, z + 1, min_height, max_height);

                const glm::vec3 n00 = compute_vertex_normal(x, z);
                const glm::vec3 n10 = compute_vertex_normal(x + 1, z);
                const glm::vec3 n01 = compute_vertex_normal(x, z + 1);
                const glm::vec3 n11 = compute_vertex_normal(x + 1, z + 1);

                append_triangle(p00, n00, p11, n11, p10, n10);
                append_triangle(p00, n00, p01, n01, p11, n11);
            }
        }
    }

    std::shared_ptr<Shape> HeightMapItem::get_shape() {
        return std::make_shared<FromMeshShape>(shape_id, vertices, normals);
    }

    JPH::Body *HeightMapItem::get_body() { return body; }

    glm::vec3 HeightMapItem::_get_scale() { return {1.f, 1.f, 1.f}; }

    HeightMapItem::~HeightMapItem() { image_grey.clear(); }

}// namespace arenai::model
