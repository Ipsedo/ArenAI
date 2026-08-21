//
// Created by samuel on 23/03/2023.
//

#include "./convex.h"

#include <algorithm>
#include <limits>
#include <utility>

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>

#include "../jolt_engine.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    ConvexItem::ConvexItem(
        std::string name, JoltPhysicEngine &engine, const std::shared_ptr<Shape> &shape,
        const glm::vec3 position, const glm::vec3 scale, const float mass, const glm::quat rotation)
        : JoltItem(std::move(name), engine), shape(shape), scale(scale) {

        // scale baked into the points, like Bullet's local scaling on the hull
        JPH::Array<JPH::Vec3> points;
        glm::vec3 aabb_min(std::numeric_limits<float>::infinity());
        glm::vec3 aabb_max(-std::numeric_limits<float>::infinity());
        for (auto [x, y, z]: shape->get_vertices()) {
            const glm::vec3 point = glm::vec3(x, y, z) * scale;
            points.emplace_back(point.x, point.y, point.z);

            aabb_min = glm::min(aabb_min, point);
            aabb_max = glm::max(aabb_max, point);
        }

        const JPH::ConvexHullShapeSettings hull_settings(points);
        JPH::ShapeRefC hull_shape = hull_settings.Create().Get();

        // Bullet rotates bodies about the mesh origin, not the hull centroid:
        // pull the center of mass back onto the origin
        const JPH::OffsetCenterOfMassShapeSettings com_settings(
            -hull_shape->GetCenterOfMass(), hull_shape);
        const JPH::ShapeRefC body_shape = com_settings.Create().Get();

        JPH::BodyCreationSettings body_settings(
            body_shape, JPH::RVec3(position.x, position.y, position.z),
            JPH::Quat(rotation.x, rotation.y, rotation.z, rotation.w),
            mass == 0.f ? JPH::EMotionType::Static : JPH::EMotionType::Dynamic,
            mass == 0.f ? layers::NON_MOVING : layers::MOVING);

        // Bullet defaults: friction 0.5, no restitution, no damping
        body_settings.mFriction = 0.5f;
        body_settings.mRestitution = 0.f;
        body_settings.mLinearDamping = 0.f;
        body_settings.mAngularDamping = 0.f;

        if (mass != 0.f) {
            // Bullet's convex hulls use the inertia of their margin-inflated AABB
            constexpr float margin = 0.04f;
            const glm::vec3 half_extents = (aabb_max - aabb_min) * 0.5f + margin;
            const glm::vec3 full_extents = 2.f * half_extents;

            body_settings.mOverrideMassProperties =
                JPH::EOverrideMassProperties::MassAndInertiaProvided;
            body_settings.mMassPropertiesOverride.mMass = mass;
            body_settings.mMassPropertiesOverride.mInertia = JPH::Mat44::sScale(JPH::Vec3(
                mass / 12.f * (full_extents.y * full_extents.y + full_extents.z * full_extents.z),
                mass / 12.f * (full_extents.x * full_extents.x + full_extents.z * full_extents.z),
                mass / 12.f * (full_extents.x * full_extents.x + full_extents.y * full_extents.y)));
        }

        body = engine.get_body_interface().CreateBody(body_settings);
        body->SetUserData(reinterpret_cast<JPH::uint64>(static_cast<Item *>(this)));
    }

    std::shared_ptr<Shape> ConvexItem::get_shape() { return shape; }

    JPH::Body *ConvexItem::get_body() { return body; }

    glm::vec3 ConvexItem::_get_scale() { return scale; }

    /*
 * Basic shapes
 */

    CubeItem::CubeItem(
        std::string name, JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const glm::vec3 position, const glm::vec3 scale, const float mass)
        : ConvexItem(
            std::move(name), engine,
            std::make_shared<ObjShape>(file_reader, std::filesystem::path("obj") / "cube.obj"),
            position, scale, mass) {}

    SphereItem::SphereItem(
        std::string name, JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const glm::vec3 position, const glm::vec3 scale, const float mass)
        : ConvexItem(
            std::move(name), engine,
            std::make_shared<ObjShape>(file_reader, std::filesystem::path("obj") / "sphere.obj"),
            position, scale, mass) {}

    CylinderItem::CylinderItem(
        std::string name, JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const glm::vec3 position, const glm::vec3 scale, const float mass)
        : ConvexItem(
            std::move(name), engine,
            std::make_shared<ObjShape>(file_reader, std::filesystem::path("obj") / "cylinder.obj"),
            position, scale, mass) {}

    TetraItem::TetraItem(
        std::string name, JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const glm::vec3 position, const glm::vec3 scale, const float mass)
        : ConvexItem(
            std::move(name), engine,
            std::make_shared<ObjShape>(file_reader, std::filesystem::path("obj") / "tetra.obj"),
            position, scale, mass) {}

}// namespace arenai::model
