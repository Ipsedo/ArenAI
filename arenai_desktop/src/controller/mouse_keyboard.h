//
// Created by samuel on 16/03/2026.
//

#ifndef ARENAI_DESKTOP_PLAYER_CONTROLLER_HANDLER_H
#define ARENAI_DESKTOP_PLAYER_CONTROLLER_HANDLER_H

#include <memory>
#include <optional>
#include <utility>

#include <arenai_controller/callback.h>
#include <arenai_controller/handler.h>
#include <arenai_view/renderer.h>
#include <arenai_view/window.h>

#include "./bindings.h"

namespace arenai::desktop {

    struct PlayerMouseKeyboardInput {
        std::optional<std::pair<controller::Key, controller::InputAction>> key;
        std::optional<std::pair<controller::MouseButton, controller::InputAction>> button;
        double mouse_x{};
        double mouse_y{};
    };

    class PlayerMouseKeyboardHandler : public controller::EventHandler<PlayerMouseKeyboardInput>,
                                       public controller::AbstractKeyboardCallback {
    public:
        PlayerMouseKeyboardHandler(
            std::shared_ptr<view::AbstractWindow> window, const view::AbstractRenderer &renderer,
            const KeyboardBindings &bindings);

        void on_key(controller::Key key, controller::InputAction action) override;
        void on_mouse_move(double x, double y) override;
        void
        on_mouse_button(controller::MouseButton button, controller::InputAction action) override;

    protected:
        std::tuple<bool, controller::user_input> to_output(PlayerMouseKeyboardInput event) override;

    private:
        // presses / releases of a bound key or mouse button drive the
        // movement state and the fire flag; an unbound slot never matches
        void apply_binding(
            const KeyboardBinding &input, controller::InputAction action, bool &need_fire);

        std::shared_ptr<view::AbstractWindow> window;

        const view::AbstractRenderer &renderer;

        KeyboardBindings bindings;

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
