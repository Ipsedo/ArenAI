//
// Created by samuel on 02/04/2023.
//

#include "./canon.h"

#include <glm/gtc/type_ptr.hpp>

#include "./shell.h"

CanonItem::CanonItem(
    const std::string &prefix_name, const std::shared_ptr<AbstractFileReader> &file_reader,
    glm::vec3 pos, glm::vec3 rel_pos, glm::vec3 scale, float mass, btRigidBody *turret)
    : ConvexItem(
        prefix_name + "_canon", std::make_shared<ObjShape>(file_reader, "obj/anubis_canon.obj"),
        pos, scale, mass),
      angle(0.f), file_reader(file_reader), will_fire(false) {

    btVector3 turret_pivot = btVector3(rel_pos.x, rel_pos.y, rel_pos.z);
    btVector3 canon_pivot = btVector3(0.f, 0.f, 0);
    btVector3 axis = btVector3(1, 0, 0);
    hinge = new btHingeConstraint(
        *turret, *ConvexItem::get_body(), turret_pivot, canon_pivot, axis, axis, true);

    hinge->setLimit(angle, angle);
}

std::vector<std::shared_ptr<Item>> CanonItem::get_produced_items() {

    if (will_fire) {
        btTransform canon_tr = ConvexItem::get_body()->getWorldTransform();
        float tmp[16];
        canon_tr.getOpenGLMatrix(tmp);
        glm::mat4 m_matrix = glm::make_mat4(tmp);

        glm::vec4 shell_pos(0.f, 0.f, 3.f, 1.f);
        shell_pos = m_matrix * shell_pos;

        will_fire = false;

        auto shell_item = std::make_shared<ShellItem>(
            file_reader, glm::vec3(shell_pos), glm::toQuat(m_matrix), glm::vec3(0.2f), 20.f);

        glm::vec4 force_vec(0.f, 0.f, 1.f, 0.f);
        force_vec = m_matrix * force_vec;

        shell_item->get_body()->applyCentralForce(
            btVector3(force_vec.x, force_vec.y, force_vec.z) * 1.5e5f);

        return {shell_item};
    }

    return {};
}

void CanonItem::on_input(const user_input &input) {
    angle += input.right_joystick.y * 2.f;

    angle = angle > 1.f ? 1.f : angle;
    angle = angle < -1.f ? -1.f : angle;

    hinge->setLimit(angle * float(M_PI) * 0.2f, angle * float(M_PI) * 0.2f);

    if (input.fire_button.pressed) will_fire = true;
}

glm::vec3 CanonItem::pos() {
    btTransform tr = ConvexItem::get_body()->getWorldTransform();
    float tmp[16];
    tr.getOpenGLMatrix(tmp);
    glm::mat4 model_mat = glm::make_mat4(tmp);

    return model_mat * glm::vec4(0, 4, -20, 1);
}

glm::vec3 CanonItem::look() {
    btTransform tr = ConvexItem::get_body()->getWorldTransform();
    float tmp[16];
    tr.getOpenGLMatrix(tmp);
    glm::mat4 model_mat = glm::make_mat4(tmp);

    return model_mat * glm::vec4(0, 0, 1, 1);
}

glm::vec3 CanonItem::up() {
    btTransform tr = ConvexItem::get_body()->getWorldTransform();
    float tmp[16];
    tr.getOpenGLMatrix(tmp);
    glm::mat4 model_mat = glm::make_mat4(tmp);

    return model_mat * glm::vec4(0, 1, 0, 0);
}

std::vector<btTypedConstraint *> CanonItem::get_constraints() {
    auto constraints = Item::get_constraints();
    constraints.push_back(hinge);
    return constraints;
}
