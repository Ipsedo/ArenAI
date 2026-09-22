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
// Helpers — the handler integrates the stick into an absolute aim target
// ========================================================================

namespace {

    // the gains held by PlayerGamepadHandler::to_output
    constexpr float TURRET_GAIN = 0.02f;
    constexpr float CANON_GAIN = 0.04f;

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
// The stick stays a rate: deflection integrates into the held target
// ========================================================================

TEST_F(PlayerAimTest, OneTickMovesTheAimByOneGain) {
    const auto rig = make_rig();

    rig.tick(1., 0.);

    ASSERT_FLOAT_EQ(rig.aim().x, -TURRET_GAIN);
}

TEST_F(PlayerAimTest, TicksAccumulateIntoTheHeldTarget) {
    const auto rig = make_rig();

    for (int i = 0; i < 10; i++) rig.tick(1., 0.);

    ASSERT_NEAR(rig.aim().x, -10.f * TURRET_GAIN, 1e-5f);
}

TEST_F(PlayerAimTest, CenteredStickHoldsTheAimStill) {
    const auto rig = make_rig();

    for (int i = 0; i < 10; i++) rig.tick(1., 0.);
    const float held = rig.aim().x;

    for (int i = 0; i < 10; i++) rig.tick(0., 0.);

    ASSERT_FLOAT_EQ(rig.aim().x, held) << "a centered stick must not move the aim";
}

// ========================================================================
// Button events carry no aim: they must re-emit the held target
// ========================================================================

TEST_F(PlayerAimTest, ButtonEventKeepsTheHeldAim) {
    const auto rig = make_rig();

    for (int i = 0; i < 10; i++) rig.tick(1., 0.);
    const float held = rig.aim().x;

    rig.handler->on_gamepad_button(controller::GamepadButton::RB, controller::InputAction::Press);

    ASSERT_TRUE(rig.controller->last_input.fire_button.pressed);
    ASSERT_FLOAT_EQ(rig.aim().x, held) << "firing must not snap the turret back to dead ahead";
}

// ========================================================================
// Anti-windup — the held target never runs past what the parts accept
// ========================================================================

TEST_F(PlayerAimTest, CanonTargetSaturatesWithoutWindup) {
    const auto rig = make_rig();

    for (int i = 0; i < 100; i++) rig.tick(0., 1.);
    ASSERT_FLOAT_EQ(rig.aim().y, 1.f);

    rig.tick(0., -1.);

    ASSERT_NEAR(rig.aim().y, 1.f - CANON_GAIN, 1e-5f)
        << "the aim must come back immediately, not after unwinding";
}
