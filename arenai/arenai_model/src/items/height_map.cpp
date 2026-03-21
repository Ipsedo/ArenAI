//
// Created by samuel on 23/03/2023.
//

#include <algorithm>

#include <arenai_model/height_map.h>

HeightMapItem::HeightMapItem(
    std::string name, const std::shared_ptr<AbstractFileReader> &img_reader,
    const std::string &height_map_file, glm::vec3 pos, glm::vec3 scale)
    : Item(std::move(name)), shape_id(height_map_file), scale(scale) {
    ImageChannels tmp = img_reader->read_png(height_map_file);
    auto [width, height, pixels] = AbstractFileReader::to_img_grey(tmp);

    map_width = width;
    map_height = height;

    image_grey = pixels;

    float min_height = std::numeric_limits<float>::infinity();
    float max_height = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < width * height; i++) {
        min_height = std::min(image_grey[i], min_height);
        max_height = std::max(image_grey[i], max_height);
    }

    map = new btHeightfieldTerrainShape(
        width, height, image_grey.data(), 1.f, min_height, max_height, 1, PHY_FLOAT, false);
    map->setLocalScaling(btVector3(scale.x, scale.y, scale.z));

    build_render_mesh(
        btVector3(-2000., -2000., -2000.), btVector3(2000., 2000., 2000.), min_height, max_height);

    map->setUseDiamondSubdivision(true);

    btTransform myTransform;
    myTransform.setIdentity();
    myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));

    btVector3 intertie(0.f, 0.f, 0.f);
    auto *motionState = new btDefaultMotionState(myTransform);
    btRigidBody::btRigidBodyConstructionInfo info(0.f, motionState, map, intertie);

    body = new btRigidBody(info);
    body->setUserPointer(this);
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
    const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec3 &p3, const btVector3 &aabbMin,
    const btVector3 &aabbMax) {
    const float triMinX = std::min({p1.x, p2.x, p3.x});
    const float triMinY = std::min({p1.y, p2.y, p3.y});
    const float triMinZ = std::min({p1.z, p2.z, p3.z});

    const float triMaxX = std::max({p1.x, p2.x, p3.x});
    const float triMaxY = std::max({p1.y, p2.y, p3.y});
    const float triMaxZ = std::max({p1.z, p2.z, p3.z});

    if (triMaxX < aabbMin.x() || triMinX > aabbMax.x()) return false;
    if (triMaxY < aabbMin.y() || triMinY > aabbMax.y()) return false;
    if (triMaxZ < aabbMin.z() || triMinZ > aabbMax.z()) return false;

    return true;
}

void HeightMapItem::build_render_mesh(
    const btVector3 &aabbMin, const btVector3 &aabbMax, const float min_height,
    const float max_height) {
    vertices.clear();
    normals.clear();

    const float half_width = static_cast<float>(map_width - 1) * 0.5f;
    const float half_length = static_cast<float>(map_height - 1) * 0.5f;
    const float center_y = 0.5f * (min_height + max_height);

    auto make_pos = [this, half_width, half_length,
                     center_y](const int x, const int z) -> glm::vec3 {
        return {
            (static_cast<float>(x) - half_width) * scale.x, (get_height(x, z) - center_y) * scale.y,
            (static_cast<float>(z) - half_length) * scale.z};
    };

    auto append_triangle = [this, &aabbMin, &aabbMax](
                               const glm::vec3 &p1, const glm::vec3 &n1, const glm::vec3 &p2,
                               const glm::vec3 &n2, const glm::vec3 &p3, const glm::vec3 &n3) {
        if (!triangle_intersects_aabb(p1, p2, p3, aabbMin, aabbMax)) return;

        vertices.emplace_back(p1.x, p1.y, p1.z);
        normals.emplace_back(n1.x, n1.y, n1.z);

        vertices.emplace_back(p2.x, p2.y, p2.z);
        normals.emplace_back(n2.x, n2.y, n2.z);

        vertices.emplace_back(p3.x, p3.y, p3.z);
        normals.emplace_back(n3.x, n3.y, n3.z);
    };

    for (int z = 0; z < map_height - 1; ++z) {
        for (int x = 0; x < map_width - 1; ++x) {
            const glm::vec3 p00 = make_pos(x, z);
            const glm::vec3 p10 = make_pos(x + 1, z);
            const glm::vec3 p01 = make_pos(x, z + 1);
            const glm::vec3 p11 = make_pos(x + 1, z + 1);

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

btRigidBody *HeightMapItem::get_body() { return body; }

glm::vec3 HeightMapItem::_get_scale() { return {1.f, 1.f, 1.f}; }

HeightMapItem::~HeightMapItem() {
    delete map;
    image_grey.clear();
}
