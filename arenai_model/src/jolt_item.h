//
// Created by claude on 30/07/2026.
//

#ifndef ARENAI_JOLT_ITEM_H
#define ARENAI_JOLT_ITEM_H

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Constraints/TwoBodyConstraint.h>

#include <arenai_model/item.h>

namespace arenai::model {

    class JoltPhysicEngine;

    class JoltItem : public Item {
    public:
        JoltItem(std::string name, JoltPhysicEngine &engine);

        glm::mat4 get_model_matrix() override;

        glm::vec3 get_linear_velocity() override;
        glm::vec3 get_angular_velocity() override;

        virtual JPH::Body *get_body() = 0;
        virtual std::vector<JPH::Ref<JPH::TwoBodyConstraint>> get_constraints();

    protected:
        JoltPhysicEngine &get_engine() const;

    private:
        JoltPhysicEngine &engine;
    };

}// namespace arenai::model

#endif// ARENAI_JOLT_ITEM_H
