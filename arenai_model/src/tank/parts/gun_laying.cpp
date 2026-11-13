#include "./gun_laying.h"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <glm/gtc/quaternion.hpp>

using namespace arenai;
using namespace arenai::model;

namespace {

    constexpr float TWO_PI = 2.f * std::numbers::pi_v<float>;

    glm::vec3 from_angles(const float yaw, const float elevation) {
        return {
            std::cos(elevation) * std::sin(yaw), std::sin(elevation),
            std::cos(elevation) * std::cos(yaw)};
    }

}// namespace

namespace arenai::model {

    glm::mat3 rotation_of(const JPH::Body &body) {
        const JPH::Quat rotation = body.GetRotation();
        return glm::mat3_cast(
            glm::quat(rotation.GetW(), rotation.GetX(), rotation.GetY(), rotation.GetZ()));
    }

    GunLaying::GunLaying() : yaw(0.f), elevation(0.f) {}

    void GunLaying::rotate_turret(const float delta) { yaw = std::remainder(yaw + delta, TWO_PI); }

    void GunLaying::rotate_canon(const float delta) {
        elevation = std::clamp(
            elevation - delta, -std::numbers::pi_v<float> / 2.f, std::numbers::pi_v<float> / 2.f);
    }

    float GunLaying::turret_angle(const glm::mat3 &chassis_rotation) const {
        const glm::vec3 local = glm::transpose(chassis_rotation) * direction();
        return std::atan2(local.x, local.z);
    }

    float GunLaying::canon_angle(const glm::mat3 &turret_rotation) const {
        const glm::vec3 local = glm::transpose(turret_rotation) * direction();
        return -std::asin(std::clamp(local.y, -1.f, 1.f));
    }

    void GunLaying::anchor_turret(const glm::mat3 &chassis_rotation, const float turret_angle) {
        const glm::vec3 local = glm::transpose(chassis_rotation) * direction();
        const float horizontal = std::sqrt(local.x * local.x + local.z * local.z);
        set_direction(
            chassis_rotation
            * glm::vec3(
                horizontal * std::sin(turret_angle), local.y, horizontal * std::cos(turret_angle)));
    }

    void GunLaying::anchor_canon(const glm::mat3 &turret_rotation, const float canon_angle) {
        const glm::vec3 local = glm::transpose(turret_rotation) * direction();
        set_direction(turret_rotation * from_angles(std::atan2(local.x, local.z), -canon_angle));
    }

    void GunLaying::reset(const glm::vec3 &canon_forward) { set_direction(canon_forward); }

    glm::vec3 GunLaying::direction() const { return from_angles(yaw, elevation); }

    void GunLaying::set_direction(const glm::vec3 &world_direction) {
        const glm::vec3 unit = glm::normalize(world_direction);
        yaw = std::atan2(unit.x, unit.z);
        elevation = std::asin(std::clamp(unit.y, -1.f, 1.f));
    }

}// namespace arenai::model
