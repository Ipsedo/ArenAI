//
// Created by samuel on 02/04/2023.
//

#ifndef ARENAI_WHEEL_H
#define ARENAI_WHEEL_H

#include <cmath>

#include <arenai_controller/controller.h>
#include <arenai_controller/inputs.h>

#include "../../items/convex.h"

namespace arenai::model {

    constexpr float WHEEL_DIRECTION_MAX_RADIAN = static_cast<float>(M_PI) / 6.f;

    class WheelItem : public LifeItem, public ConvexItem, public controller::Controller {
    public:
        WheelItem(
            const std::string &prefix_name,
            const std::shared_ptr<utils::AbstractFileReader> &file_reader, glm::vec3 pos,
            glm::vec3 rel_pos, glm::vec3 scale, float mass, btRigidBody *chassis,
            float front_axle_z);
        void on_input(const controller::user_input &input) override;

        std::vector<btTypedConstraint *> get_constraints() override;

    protected:
        btGeneric6DofSpring2Constraint *hinge;

    private:
        float front_axle_z;
        glm::vec3 wheel_center_pos_rel_to_chassis{};

        float adjust_rotation_velocity_differential(
            float front_wheel_orientation_radian, float original_rotation_velocity) const;
    };

    class DirectionalWheelItem final : public WheelItem {
    public:
        DirectionalWheelItem(
            const std::string &name, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            glm::vec3 pos, glm::vec3 rel_pos, glm::vec3 scale, float mass, btRigidBody *chassis,
            float front_axle_z, float angle_factor);
        void on_input(const controller::user_input &input) override;

    private:
        float angle_factor;
    };

}// namespace arenai::model

#endif// ARENAI_WHEEL_H
