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
    const float wanted_frame_frequency, const std::function<void(Item *)> &contact_callback)
    : LifeItem(2), ConvexItem(NAME, load_shape(file_reader), pos, scale, mass),
      contact_callback(contact_callback),
      nb_frames_alive(static_cast<int>(20.f / wanted_frame_frequency)) {

    btTransform shell_tr;
    shell_tr.setIdentity();
    shell_tr.setOrigin(btVector3(pos.x, pos.y, pos.z));
    shell_tr.setRotation(btQuaternion(rot.x, rot.y, rot.z, rot.w));

    ConvexItem::get_body()->setWorldTransform(shell_tr);
}

void ShellItem::on_contact(Item *other) {
    if (const auto t = dynamic_cast<LifeItem *>(other)) t->receive_damages(1);
    receive_damages(1);

    Item::on_contact(other);
    contact_callback(other);

    if (is_dead()) destroy();
}

void ShellItem::tick() {
    ConvexItem::tick();

    nb_frames_alive--;

    if (nb_frames_alive <= 0) destroy();
}
