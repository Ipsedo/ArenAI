//
// Created by samuel on 26/03/2023.
//

#ifndef ARENAI_HUD_DRAWABLES_H
#define ARENAI_HUD_DRAWABLES_H

#include <functional>
#include <memory>

#include <arenai_controller/inputs.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/hud.h>

#include "../program.h"

namespace arenai::view {

    class ButtonDrawable final : public AbstractHudDrawable {
    public:
        ButtonDrawable(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            std::function<controller::button(void)> get_input, glm::vec2 center_px, float size_px);

        void draw(int width, int height) override;

    private:
        std::function<controller::button(void)> get_input;

        std::unique_ptr<Program> program;

        float center_x, center_y;
        float size;

        int nb_points;
    };

    class JoyStickDrawable final : public AbstractHudDrawable {
    public:
        JoyStickDrawable(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            std::function<controller::joystick(void)> get_input_px, glm::vec2 center_px,
            float size_px, float stick_size_px);

        void draw(int width, int height) override;

    private:
        std::function<controller::joystick(void)> get_input;

        std::unique_ptr<Program> program;

        float center_x, center_y;
        float size, stick_size;

        int nb_point_bound, nb_point_stick;
    };

}// namespace arenai::view

#endif// ARENAI_HUD_DRAWABLES_H
