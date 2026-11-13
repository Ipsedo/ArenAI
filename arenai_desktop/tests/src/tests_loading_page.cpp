#include <gtest/gtest.h>
#include <gui/rml/loading_page.h>

#include <arenai_desktop_tests/headless_rml.h>

using namespace arenai::desktop;

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
