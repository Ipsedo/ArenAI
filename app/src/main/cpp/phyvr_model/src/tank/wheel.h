//
// Created by samuel on 02/04/2023.
//

#ifndef PHYVR_WHEEL_H
#define PHYVR_WHEEL_H

#include <phyvr_controller/controller.h>
#include <phyvr_controller/inputs.h>
#include <phyvr_model/convex.h>

class WheelItem : public LifeItem, public ConvexItem, public Controller {
public:
    WheelItem(
        const std::string &prefix_name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 pos, glm::vec3 rel_pos, glm::vec3 scale, float mass, btRigidBody *chassis);
    void on_input(const user_input &input) override;

    std::vector<btTypedConstraint *> get_constraints() override;

protected:
    btGeneric6DofSpring2Constraint *hinge;
};

class DirectionalWheelItem : public WheelItem {
public:
    DirectionalWheelItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &file_reader, glm::vec3 pos,
        glm::vec3 rel_pos, glm::vec3 scale, float mass, btRigidBody *chassis);
    void on_input(const user_input &input) override;
};

#endif// PHYVR_WHEEL_H
