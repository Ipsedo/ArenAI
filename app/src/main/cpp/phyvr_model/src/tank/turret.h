//
// Created by samuel on 02/04/2023.
//

#ifndef PHYVR_TURRET_H
#define PHYVR_TURRET_H

#include <phyvr_controller/controller.h>
#include <phyvr_controller/inputs.h>
#include <phyvr_model/convex.h>

class TurretItem : public LifeItem, public ConvexItem, public Controller {
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

#endif// PHYVR_TURRET_H
