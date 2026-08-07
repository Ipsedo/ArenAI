//
// Created by samuel on 13/04/2023.
//

#include "./shell.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    std::shared_ptr<Shape>
    ShellItem::load_shape(const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader) {
        return std::make_shared<ObjShape>(file_reader, std::filesystem::path("obj") / "cone.obj");
    }

    ShellItem::ShellItem(
        JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader, const glm::vec3 pos,
        const glm::quat rot, const glm::vec3 scale, const float mass,
        const float wanted_frame_frequency,
        const std::function<void(glm::vec3, glm::vec3, Item *)> &contact_callback)
        : LifeItem(1), ConvexItem(NAME, engine, load_shape(file_reader), pos, scale, mass, rot),
          contact_callback(contact_callback),
          nb_frames_alive(static_cast<int>(20.f / wanted_frame_frequency)), start_pos(pos) {}

    void ShellItem::on_contact(Item *other) {
        if (const auto t = dynamic_cast<LifeItem *>(other)) t->receive_damages(1);
        receive_damages(1);

        Item::on_contact(other);
        contact_callback(get_fire_position(), get_current_position(), other);

        if (is_dead()) destroy();
    }

    void ShellItem::tick() {
        ConvexItem::tick();

        nb_frames_alive--;

        if (nb_frames_alive <= 0) destroy();
    }

    glm::vec3 ShellItem::get_fire_position() const { return start_pos; }

    glm::vec3 ShellItem::get_current_position() {
        return {get_model_matrix() * glm::vec4(glm::vec3(0.f), 1.f)};
    }

}// namespace arenai::model
