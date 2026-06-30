//
// Created by samuel on 13/04/2023.
//

#ifndef ARENAI_SHELL_H
#define ARENAI_SHELL_H

#include <functional>

#include <arenai_utils/file_reader.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include "../items/convex.h"

class ShellItem final : public LifeItem, public ConvexItem {
public:
    static std::shared_ptr<Shape>
    load_shape(const std::shared_ptr<AbstractFileReader> &file_reader);

    ShellItem(
        const std::shared_ptr<AbstractFileReader> &file_reader, glm::vec3 pos, glm::quat rot,
        glm::vec3 scale, float mass, float wanted_frame_frequency,
        const std::function<void(glm::vec3, glm::vec3, Item *)> &contact_callback =
            [](glm::vec3, glm::vec3, Item *) {});

    void on_contact(Item *other) override;

    inline const static std::string NAME = "shell_item";

    void tick() override;

    glm::vec3 get_fire_position() const;
    glm::vec3 get_current_position();

private:
    std::function<void(glm::vec3, glm::vec3, Item *)> contact_callback;
    int nb_frames_alive;

    glm::vec3 start_pos;
};

#endif// ARENAI_SHELL_H
