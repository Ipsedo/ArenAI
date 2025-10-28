//
// Created by samuel on 02/04/2023.
//

#ifndef ARENAI_CANON_H
#define ARENAI_CANON_H

#include <functional>

#include <arenai_controller/controller.h>
#include <arenai_controller/inputs.h>
#include <arenai_model/convex.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/camera.h>

class CanonItem final : public LifeItem,
                        public ConvexItem,
                        public ItemProducer,
                        public Controller,
                        public Camera {
public:
    CanonItem(
        const std::string &prefix_name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 pos, glm::vec3 rel_pos, glm::vec3 scale, float mass, btRigidBody *turret,
        float wanted_frame_frequency, const std::function<void(Item *)> &on_contact);

    void on_input(const user_input &input) override;

    glm::vec3 pos() override;
    glm::vec3 look() override;
    glm::vec3 up() override;

    std::vector<btTypedConstraint *> get_constraints() override;

    std::vector<std::shared_ptr<Item>> get_produced_items() override;

private:
    float angle;
    btHingeConstraint *hinge;
    std::shared_ptr<AbstractFileReader> file_reader;
    bool will_fire;
    std::function<void(Item *)> on_contact;
    float wanted_frame_frequency;
};

#endif// ARENAI_CANON_H
