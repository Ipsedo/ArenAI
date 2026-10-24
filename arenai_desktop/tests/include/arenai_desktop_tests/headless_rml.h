//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_TESTS_HEADLESS_RML_H
#define ARENAI_DESKTOP_TESTS_HEADLESS_RML_H

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <gui/rml/adapters.h>
#include <gui/rml/decorators/decorator_registry.h>
#include <RmlUi/Core.h>

#include <arenai_controller/input_names.h>
#include <arenai_view/window.h>

// RmlUi without a GPU: the menu documents load, lay out and answer events
// against a render interface that compiles nothing. One RmlUi lifetime per
// test binary (Rml::Initialise / Shutdown are process-wide), one context per
// test.

class NoopRenderInterface final : public Rml::RenderInterface {
public:
    Rml::CompiledGeometryHandle
    CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override {
        return 1;
    }
    void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
    void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}

    Rml::TextureHandle LoadTexture(Rml::Vector2i &, const Rml::String &) override { return 0; }
    Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override {
        return 1;
    }
    void ReleaseTexture(Rml::TextureHandle) override {}

    void EnableScissorRegion(bool) override {}
    void SetScissorRegion(Rml::Rectanglei) override {}
};

// a window the pages can talk to; records what they asked
class FakeWindow final : public arenai::view::AbstractWindow {
public:
    bool should_close() override { return false; }
    void poll_events() override {}

    void set_keyboard_callback(
        const std::shared_ptr<arenai::controller::AbstractKeyboardCallback> &) override {}
    void set_gamepad_callback(
        const std::shared_ptr<arenai::controller::AbstractGamepadCallback> &) override {}
    void set_resize_callback(std::function<void(int, int)>) override {}

    void set_cursor_mode(arenai::controller::CursorMode) override {}
    void set_cursor_position(double, double) override {}

    void set_fullscreen(const bool fullscreen) override { this->fullscreen = fullscreen; }
    std::tuple<int, int> screen_size() const override { return {1920, 1080}; }

    std::vector<arenai::view::GamepadInfo> list_gamepads() const override { return gamepads; }
    void select_gamepad(const int id) override { selected_gamepad = id; }

    std::string key_label(const arenai::controller::Key key) const override {
        return arenai::controller::to_string(key);
    }

    bool fullscreen = false;
    std::vector<arenai::view::GamepadInfo> gamepads;
    int selected_gamepad = -2;
};

// process-wide RmlUi lifetime, registered once with gtest
class HeadlessRmlEnvironment final : public testing::Environment {
public:
    void SetUp() override;
    void TearDown() override;

private:
    arenai::desktop::gui::MenuSystemInterface system_interface_;
    NoopRenderInterface render_interface_;
    std::unique_ptr<arenai::desktop::gui::ReaderBackedFileInterface> file_interface_;
    std::unique_ptr<arenai::desktop::gui::DecoratorRegistry> decorators_;
    std::vector<std::string> font_buffers_;
};

// one RmlUi context per test, named after the test
class HeadlessRmlTest : public testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    Rml::Context &context() const { return *context_; }

    // the first descendant whose data-event-click reads exactly `handler`
    static Rml::Element *find_event_element(Rml::Element &root, const Rml::String &handler);

private:
    std::string context_name_;
    Rml::Context *context_ = nullptr;
};

#endif// ARENAI_DESKTOP_TESTS_HEADLESS_RML_H
