//
// Created by samuel on 02/04/2023.
//

#ifndef ARENAI_CANON_H
#define ARENAI_CANON_H

#include <functional>

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>

#include <arenai_controller/controller.h>
#include <arenai_controller/inputs.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/camera.h>

#include "./items/convex.h"
#include "./shell.h"

namespace arenai::model {

    class CanonItem final : public LifeItem,
                            public ConvexItem,
                            public ItemProducer,
                            public controller::Controller,
                            public view::AbstractCamera {
    public:
        CanonItem(
            const std::string &prefix_name, JoltPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader, glm::vec3 pos,
            glm::vec3 rel_pos, glm::vec3 scale, float mass, JPH::Body *turret,
            float wanted_frame_frequency,
            const std::function<void(glm::vec3, glm::vec3, Item *)> &on_contact,
            const std::function<void(const std::shared_ptr<ShellItem> &)> &on_shell_fired,
            const std::function<bool()> &can_fire);

        void apply_input(const controller::user_input &input) override;

        glm::vec3 pos() override;
        glm::vec3 look() override;
        glm::vec3 up() override;

        std::vector<JPH::Ref<JPH::TwoBodyConstraint>> get_constraints() override;

        std::vector<std::shared_ptr<Item>> get_produced_items() override;
        std::vector<std::shared_ptr<JoltItem>> produce_jolt_items();

    private:
        float angle;
        JPH::Ref<JPH::HingeConstraint> hinge;
        std::shared_ptr<utils::AbstractResourceFileReader> file_reader;
        bool will_fire;
        std::function<void(glm::vec3, glm::vec3, Item *)> on_contact;
        std::function<void(const std::shared_ptr<ShellItem> &)> on_shell_fired;
        std::function<bool()> can_fire;
        float wanted_frame_frequency;
    };

}// namespace arenai::model

#endif// ARENAI_CANON_H
