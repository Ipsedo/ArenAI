//
// Created by samuel on 23/03/2023.
//

#ifndef ARENAI_CONVEX_H
#define ARENAI_CONVEX_H

#include <glm/gtc/quaternion.hpp>
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/Body.h>

#include <arenai_model/item.h>
#include <arenai_utils/file_reader.h>

#include "../jolt_item.h"

namespace arenai::model {

    class ConvexItem : public JoltItem {
    public:
        ConvexItem(
            std::string name, JoltPhysicEngine &engine, const std::shared_ptr<Shape> &shape,
            glm::vec3 position, glm::vec3 scale, float mass,
            glm::quat rotation = glm::quat(1.f, 0.f, 0.f, 0.f));

        std::shared_ptr<Shape> get_shape() override;

        JPH::Body *get_body() override;

    protected:
        glm::vec3 _get_scale() override;

    private:
        JPH::Body *body;

        std::shared_ptr<Shape> shape;
        glm::vec3 scale;
    };

    class SphereItem final : public ConvexItem {
    public:
        SphereItem(
            std::string name, JoltPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass);
    };

    class CubeItem final : public ConvexItem {
    public:
        CubeItem(
            std::string name, JoltPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass);
    };

    class CylinderItem final : public ConvexItem {
    public:
        CylinderItem(
            std::string name, JoltPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass);
    };

    class TetraItem final : public ConvexItem {
    public:
        TetraItem(
            std::string name, JoltPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass);
    };

}// namespace arenai::model

#endif// ARENAI_CONVEX_H
