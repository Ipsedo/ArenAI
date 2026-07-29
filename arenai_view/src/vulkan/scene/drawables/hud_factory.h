//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_HUD_FACTORY_H
#define ARENAI_VK_HUD_FACTORY_H

#include <arenai_view/hud.h>

namespace arenai::view {

    class VulkanHudFactory final : public AbstractHudFactory {
    public:
        std::unique_ptr<AbstractHudDrawable> make_joystick(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            std::function<controller::joystick(void)> get_input_px, glm::vec2 center_px,
            float size_px, float stick_size_px) override;

        std::unique_ptr<AbstractHudDrawable> make_button(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            std::function<controller::button(void)> get_input, glm::vec2 center_px,
            float size_px) override;
    };

}// namespace arenai::view

#endif// ARENAI_VK_HUD_FACTORY_H
