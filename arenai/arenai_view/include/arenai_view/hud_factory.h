//
// Created by samuel on 03/07/2026.
//

#ifndef ARENAI_HUD_FACTORY_H
#define ARENAI_HUD_FACTORY_H
#include <memory>

#include "./hud.h"

namespace arenai::view {

    class HudFactory {
    public:
        virtual ~HudFactory() = default;

        virtual std::shared_ptr<HUDDrawable> make_joystick(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            std::function<controller::joystick(void)> get_input_px, glm::vec2 center_px,
            float size_px, float stick_size_px) = 0;

        virtual std::shared_ptr<HUDDrawable> make_button(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            std::function<controller::button(void)> get_input, glm::vec2 center_px,
            float size_px) = 0;
    };

}// namespace arenai::view

#endif//ARENAI_HUD_FACTORY_H
