//
// Created by samuel on 27/08/2026.
//

#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <gui/rml/input.h>

using namespace arenai;
using namespace arenai::desktop;

// ========================================================================
// MenuInputAdapter binding capture: the controls page uninstalls the sink
// (set_capture_sink(nullptr)) from inside the sink itself as soon as an
// event completes the capture. A joystick event carries two axis values
// (X then Y), so the adapter must survive the sink dying on the first one
// and drop the second instead of invoking an empty std::function.
// ========================================================================

namespace {

    // no Rml::Context needed: the capture path never touches it
    gui::MenuInputAdapter make_adapter() {
        return gui::MenuInputAdapter(
            nullptr, [] {}, [](bool) {});
    }

}// namespace

TEST(MenuInputCaptureTest, joystick_capture_completing_on_x_drops_y) {
    auto adapter = make_adapter();

    std::vector<gui::RawMenuInput> received;
    adapter.set_capture_sink([&](const gui::RawMenuInput &input) {
        received.push_back(input);
        // what assign_gamepad_axis does when the deflection binds the slot
        adapter.set_capture_sink(nullptr);
    });

    // full steer deflection on the left stick: X completes the capture
    adapter.on_joystick(1.0, 0.0, controller::GamepadJoystick::Left);

    ASSERT_EQ(received.size(), 1u);
    const auto *motion = std::get_if<std::pair<GamepadAxis, double>>(&received.front());
    ASSERT_NE(motion, nullptr);
    EXPECT_EQ(motion->first, GamepadAxis::LeftStickX);
    EXPECT_DOUBLE_EQ(motion->second, 1.0);
}

TEST(MenuInputCaptureTest, joystick_capture_kept_installed_receives_both_axes) {
    auto adapter = make_adapter();

    std::vector<gui::RawMenuInput> received;
    adapter.set_capture_sink([&](const gui::RawMenuInput &input) { received.push_back(input); });

    // small deflection: the capture stays armed, both axes flow through
    adapter.on_joystick(0.1, 0.2, controller::GamepadJoystick::Left);

    ASSERT_EQ(received.size(), 2u);
    const auto *x_motion = std::get_if<std::pair<GamepadAxis, double>>(&received[0]);
    const auto *y_motion = std::get_if<std::pair<GamepadAxis, double>>(&received[1]);
    ASSERT_NE(x_motion, nullptr);
    ASSERT_NE(y_motion, nullptr);
    EXPECT_EQ(x_motion->first, GamepadAxis::LeftStickX);
    EXPECT_EQ(y_motion->first, GamepadAxis::LeftStickY);
}
