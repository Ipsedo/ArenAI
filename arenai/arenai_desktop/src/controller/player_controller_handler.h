//
// Created by samuel on 16/03/2026.
//

#ifndef ARENAI_DESKTOP_PLAYER_CONTROLLER_HANDLER_H
#define ARENAI_DESKTOP_PLAYER_CONTROLLER_HANDLER_H

#include <memory>
#include <optional>
#include <utility>

#include <arenai_controller/handler.h>
#include <arenai_view/window.h>

namespace arenai::desktop {

    struct PlayerRawInput {
        std::optional<std::pair<view::Key, view::InputAction>> key;
        std::optional<std::pair<view::MouseButton, view::InputAction>> button;
        double mouse_x{};
        double mouse_y{};
    };

    class MouseKeyboardPlayerControllerHandler : public controller::EventHandler<PlayerRawInput>,
                                                 public view::AbstractWindowCallback {
    public:
        explicit MouseKeyboardPlayerControllerHandler(std::shared_ptr<view::AbstractWindow> window);

        void on_key(view::Key key, view::InputAction action) override;
        void on_mouse_move(double x, double y) override;
        void on_mouse_button(view::MouseButton button, view::InputAction action) override;

    protected:
        std::tuple<bool, controller::user_input> to_output(PlayerRawInput event) override;

    private:
        std::shared_ptr<view::AbstractWindow> window;

        double last_mouse_x;
        double last_mouse_y;

        float current_dir;
        float current_speed;

        float current_turret_rotation;
        float current_canon_rotation;

        bool cursor_captured;
    };

}// namespace arenai::desktop

#endif//ARENAI_DESKTOP_PLAYER_CONTROLLER_HANDLER_H
