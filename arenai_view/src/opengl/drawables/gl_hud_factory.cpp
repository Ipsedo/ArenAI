//
// Created by samuel on 08/07/2026.
//

#include "./gl_hud_factory.h"

#include "./hud_drawables.h"

namespace arenai::view {

    std::unique_ptr<AbstractHudDrawable> GlHudFactory::make_joystick(
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        std::function<controller::joystick(void)> get_input_px, glm::vec2 center_px, float size_px,
        float stick_size_px) {
        return std::make_unique<JoyStickDrawable>(
            file_reader, std::move(get_input_px), center_px, size_px, stick_size_px);
    }

    std::unique_ptr<AbstractHudDrawable> GlHudFactory::make_button(
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        std::function<controller::button(void)> get_input, glm::vec2 center_px, float size_px) {
        return std::make_unique<ButtonDrawable>(
            file_reader, std::move(get_input), center_px, size_px);
    }

}// namespace arenai::view
