//
// Created by samuel on 17/07/2026.
//

#include "./hud_factory.h"

#include "./hud_drawables.h"

namespace arenai::view {

    std::unique_ptr<AbstractHudDrawable> VulkanHudFactory::make_joystick(
        const std::shared_ptr<utils::AbstractResourceFileReader> &,
        std::function<controller::joystick(void)> get_input_px, const glm::vec2 center_px,
        const float size_px, const float stick_size_px) {
        return std::make_unique<VulkanJoyStickDrawable>(
            std::move(get_input_px), center_px, size_px, stick_size_px);
    }

    std::unique_ptr<AbstractHudDrawable> VulkanHudFactory::make_button(
        const std::shared_ptr<utils::AbstractResourceFileReader> &,
        std::function<controller::button(void)> get_input, const glm::vec2 center_px,
        const float size_px) {
        return std::make_unique<VulkanButtonDrawable>(std::move(get_input), center_px, size_px);
    }

}// namespace arenai::view
