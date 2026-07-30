//
// Created by samuel on 02/04/2023.
//

#ifndef ARENAI_TURRET_H
#define ARENAI_TURRET_H

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>

#include <arenai_controller/controller.h>
#include <arenai_controller/inputs.h>

#include "../../items/convex.h"

namespace arenai::model {

    class TurretItem final : public LifeItem, public ConvexItem, public controller::Controller {
    public:
        TurretItem(
            const std::string &prefix_name, JoltPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader, glm::vec3 pos,
            glm::vec3 rel_pos, glm::vec3 scale, float mass, JPH::Body *chassis);

        void apply_input(const controller::user_input &input) override;

        std::vector<JPH::Ref<JPH::TwoBodyConstraint>> get_constraints() override;

    private:
        float angle;
        JPH::Ref<JPH::HingeConstraint> hinge;
    };

}// namespace arenai::model

#endif// ARENAI_TURRET_H
