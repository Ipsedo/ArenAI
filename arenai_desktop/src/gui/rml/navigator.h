#ifndef ARENAI_DESKTOP_GUI_RML_NAVIGATOR_H
#define ARENAI_DESKTOP_GUI_RML_NAVIGATOR_H

#include <map>
#include <memory>
#include <vector>

#include "./menu_page.h"

namespace arenai::desktop::gui {

    class Navigator final : public AbstractNavigator {
    public:
        // loads every page into the context; throws when one fails
        Navigator(Rml::Context &context, std::vector<std::unique_ptr<MenuPage>> pages, PageId root);

        void go(PageId page) override;
        void back() override;

        void show_root();
        void hide_all();

        bool on_escape();

        // per-frame hooks forwarded to the visible page
        void update() const;
        void after_context_update() const;

        MenuPage &page(PageId page) const;
        MenuPage *current() const { return current_; }

    private:
        std::map<PageId, std::unique_ptr<MenuPage>> pages_;
        PageId root_;
        MenuPage *current_ = nullptr;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_NAVIGATOR_H
