//
// Created by samuel on 25/10/2026.
//

#include <gui/rml/fonts.h>

#include <arenai_agent/file_reader.h>
#include <arenai_desktop_tests/headless_rml.h>

using namespace arenai;
using namespace arenai::desktop;

void HeadlessRmlEnvironment::SetUp() {
    file_interface_ = std::make_unique<gui::ReaderBackedFileInterface>(
        std::make_shared<agent::DesktopAssetFileReader>(ARENAI_ASSETS_DIR));

    Rml::SetSystemInterface(&system_interface_);
    Rml::SetFileInterface(file_interface_.get());
    Rml::SetRenderInterface(&render_interface_);
    ASSERT_TRUE(Rml::Initialise());

    decorators_ = std::make_unique<gui::DecoratorRegistry>();
    gui::register_menu_decorators(*decorators_);
    font_buffers_ =
        gui::load_menu_fonts(std::make_shared<agent::DesktopAssetFileReader>(ARENAI_ASSETS_DIR));
}

void HeadlessRmlEnvironment::TearDown() {
    Rml::Shutdown();
    decorators_.reset();
    font_buffers_.clear();
}

namespace {
    // registered at static-init time, before main() runs the tests
    [[maybe_unused]] const auto *const RML_ENVIRONMENT =
        testing::AddGlobalTestEnvironment(new HeadlessRmlEnvironment);
}// namespace

void HeadlessRmlTest::SetUp() {
    const auto *info = testing::UnitTest::GetInstance()->current_test_info();
    context_name_ = std::string(info->test_suite_name()) + "." + info->name();
    context_ = Rml::CreateContext(context_name_, Rml::Vector2i(1920, 1080));
    ASSERT_NE(context_, nullptr);
}

void HeadlessRmlTest::TearDown() {
    if (context_ != nullptr) Rml::RemoveContext(context_name_);
}

Rml::Element *HeadlessRmlTest::find_event_element(Rml::Element &root, const Rml::String &handler) {
    Rml::ElementList candidates;
    root.QuerySelectorAll(candidates, "[data-event-click]");
    for (Rml::Element *element: candidates)
        if (element->GetAttribute<Rml::String>("data-event-click", "") == handler) return element;
    return nullptr;
}
