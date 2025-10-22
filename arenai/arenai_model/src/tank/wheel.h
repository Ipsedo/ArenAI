//
// Created by samuel on 02/04/2023.
//

#ifndef ARENAI_WHEEL_H
#define ARENAI_WHEEL_H

#include <arenai_controller/controller.h>
#include <arenai_controller/inputs.h>
#include <arenai_model/convex.h>

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

class DirectionalWheelItem final : public WheelItem {
public:
    DirectionalWheelItem(
        std::string name, const std::shared_ptr<AbstractFileReader> &file_reader, glm::vec3 pos,
        glm::vec3 rel_pos, glm::vec3 scale, float mass, btRigidBody *chassis);
    void on_input(const user_input &input) override;
};

#endif// ARENAI_WHEEL_H
