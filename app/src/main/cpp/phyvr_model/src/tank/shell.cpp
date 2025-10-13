//
// Created by samuel on 13/04/2023.
//

#include "./shell.h"

#include <utility>

std::shared_ptr<Shape>
ShellItem::load_shape(const std::shared_ptr<AbstractFileReader> &file_reader) {
    return std::make_shared<ObjShape>(file_reader, "obj/cone.obj");
}

ShellItem::ShellItem(
    const std::shared_ptr<AbstractFileReader> &file_reader, const glm::vec3 pos,
    const glm::quat rot, const glm::vec3 scale, const float mass,
    const std::function<void(Item *)> &contact_callback)
    : LifeItem(10),
      ConvexItem(ShellItem::NAME, ShellItem::load_shape(file_reader), pos, scale, mass),
      contact_callback(contact_callback) {

    btTransform shell_tr;
    shell_tr.setIdentity();
    shell_tr.setOrigin(btVector3(pos.x, pos.y, pos.z));
    shell_tr.setRotation(btQuaternion(rot.x, rot.y, rot.z, rot.w));

    ConvexItem::get_body()->setWorldTransform(shell_tr);
}

void ShellItem::on_contact(Item *other) {
    Item::on_contact(other);
    contact_callback(other);

    if (const auto t = dynamic_cast<LifeItem *>(other)) t->receive_damages(1);

    receive_damages(1);

    if (is_dead()) destroy();
}
