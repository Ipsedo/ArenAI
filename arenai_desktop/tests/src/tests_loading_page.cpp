//
// Created by samuel on 28/10/2026.
//

#include <gtest/gtest.h>
#include <gui/rml/loading_page.h>

#include <arenai_desktop_tests/headless_rml.h>

using namespace arenai::desktop;

// ========================================================================
// The loading document against a headless RmlUi: it loads, shows with its
// animated elements in place, and its keyframes tick without a hitch.
// ========================================================================

namespace {

    class LoadingPageTest : public HeadlessRmlTest {};

    TEST_F(LoadingPageTest, ShowsTheAnimatedDocument) {
        gui::LoadingPage page;
        page.load(context());

        page.show(Rml::ModalFlag::None, Rml::FocusFlag::None);
        context().Update();
        ASSERT_TRUE(page.visible());
        EXPECT_EQ(page.document().GetId(), "loading-body");

        Rml::ElementList diamonds;
        page.document().QuerySelectorAll(diamonds, ".loading-diamond");
        EXPECT_EQ(diamonds.size(), 4u);

        Rml::ElementList labels;
        page.document().QuerySelectorAll(labels, ".loading-label");
        EXPECT_EQ(labels.size(), 1u);

        // a few frames: the RCSS animations advance without tripping
        for (int i = 0; i < 5; ++i) context().Update();

        page.hide();
        EXPECT_FALSE(page.visible());
    }

}// namespace
