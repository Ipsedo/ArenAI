//
// Created by samuel on 26/03/2023.
//

#ifndef ARENAI_HUD_H
#define ARENAI_HUD_H

#include <functional>
#include <memory>

#include <glm/glm.hpp>

#include <arenai_controller/inputs.h>
#include <arenai_utils/file_reader.h>

namespace arenai::view {

    class AbstractHudDrawable {
    public:
        virtual void draw(int width, int height) = 0;
        virtual ~AbstractHudDrawable() = default;
    };

    class AbstractHudFactory {
    public:
        virtual ~AbstractHudFactory() = default;

        virtual std::unique_ptr<AbstractHudDrawable> make_joystick(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            std::function<controller::joystick(void)> get_input_px, glm::vec2 center_px,
            float size_px, float stick_size_px) = 0;

        virtual std::unique_ptr<AbstractHudDrawable> make_button(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            std::function<controller::button(void)> get_input, glm::vec2 center_px,
            float size_px) = 0;
    };

}// namespace arenai::view

#endif// ARENAI_HUD_H
