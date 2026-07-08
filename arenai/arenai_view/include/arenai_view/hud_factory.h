//
// Created by samuel on 03/07/2026.
//

#ifndef ARENAI_HUD_FACTORY_H
#define ARENAI_HUD_FACTORY_H

#include <functional>
#include <memory>

#include <glm/glm.hpp>

#include <arenai_controller/inputs.h>
#include <arenai_utils/file_reader.h>

#include "./hud.h"

using namespace arenai;

namespace arenai::view {

    class AbstractHudFactory {
    public:
        virtual ~AbstractHudFactory() = default;

        virtual std::unique_ptr<AbstractHudDrawable> make_joystick(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            std::function<controller::joystick(void)> get_input_px, glm::vec2 center_px,
            float size_px, float stick_size_px) = 0;

        virtual std::unique_ptr<AbstractHudDrawable> make_button(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            std::function<controller::button(void)> get_input, glm::vec2 center_px,
            float size_px) = 0;
    };

}// namespace arenai::view

#endif//ARENAI_HUD_FACTORY_H
