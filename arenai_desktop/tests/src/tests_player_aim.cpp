//
// Created by samuel on 23/09/2026.
//

#include <memory>

#include <gtest/gtest.h>

#include <arenai_controller/controller.h>

#include "controller/gamepad.h"

using namespace arenai;
using namespace arenai::desktop;

// ========================================================================
// Helpers — the handler hands the parts an aim rate, not a target
// ========================================================================

namespace {

    // the sensitivity held by PlayerGamepadHandler::to_output
    constexpr float STICK_SENSITIVITY = 0.05f;

    class RecordingController final : public controller::Controller {
    public:
        void apply_input(const controller::user_input &input) override { last_input = input; }

        controller::user_input last_input{};
    };

    struct GamepadRig {
        std::unique_ptr<PlayerGamepadHandler> handler;
        std::shared_ptr<RecordingController> controller;

        // one frame: the window dispatches the right stick last, which drives the tick
        void tick(const double x, const double y) const {
            handler->on_joystick(x, y, controller::GamepadJoystick::Right);
        }

        controller::joystick aim() const { return controller->last_input.right_joystick; }
    };

    GamepadRig make_rig() {
        auto controller = std::make_shared<RecordingController>();
        auto handler = std::make_unique<PlayerGamepadHandler>(GamepadBindings{});
        handler->add_controller(controller);

        return {std::move(handler), controller};
    }

}// namespace

class PlayerAimTest : public testing::Test {};

// ========================================================================
// The stick is a rate: full deflection asks for a share of the slew speed
// ========================================================================

TEST_F(PlayerAimTest, FullDeflectionAsksForTheStickSensitivity) {
    const auto rig = make_rig();

    rig.tick(1., 0.);

    ASSERT_FLOAT_EQ(rig.aim().x, STICK_SENSITIVITY);
}

TEST_F(PlayerAimTest, TicksDoNotAccumulate) {
    const auto rig = make_rig();

    for (int i = 0; i < 10; i++) rig.tick(1., 0.);

    ASSERT_FLOAT_EQ(rig.aim().x, STICK_SENSITIVITY) << "the stick is a rate, not a target";
}

TEST_F(PlayerAimTest, CenteredStickStopsTheAim) {
    const auto rig = make_rig();

    for (int i = 0; i < 10; i++) rig.tick(1., 0.);
    rig.tick(0., 0.);

    ASSERT_FLOAT_EQ(rig.aim().x, 0.f) << "a centered stick must ask for no rotation";
}

// ========================================================================
// Button events carry no aim
// ========================================================================

TEST_F(PlayerAimTest, ButtonEventCarriesNoAim) {
    const auto rig = make_rig();

    rig.tick(1., 0.);
    rig.handler->on_gamepad_button(controller::GamepadButton::RB, controller::InputAction::Press);

    ASSERT_TRUE(rig.controller->last_input.fire_button.pressed);
    ASSERT_FLOAT_EQ(rig.aim().x, 0.f);
}
