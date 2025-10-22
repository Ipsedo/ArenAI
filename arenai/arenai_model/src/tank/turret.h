//
// Created by samuel on 02/04/2023.
//

#ifndef ARENAI_TURRET_H
#define ARENAI_TURRET_H

#include <arenai_controller/controller.h>
#include <arenai_controller/inputs.h>
#include <arenai_model/convex.h>

class TurretItem final : public LifeItem, public ConvexItem, public Controller {
public:
    TurretItem(
        const std::string &prefix_name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 pos, glm::vec3 rel_pos, glm::vec3 scale, float mass, btRigidBody *chassis);

    void on_input(const user_input &input) override;

    std::vector<btTypedConstraint *> get_constraints() override;

private:
    float angle;
    btHingeConstraint *hinge;
};

#endif// ARENAI_TURRET_H
