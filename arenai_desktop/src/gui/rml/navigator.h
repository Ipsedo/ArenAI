//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_NAVIGATOR_H
#define ARENAI_DESKTOP_GUI_RML_NAVIGATOR_H

#include <map>
#include <memory>
#include <vector>

#include "./menu_page.h"

namespace arenai::desktop::gui {

    // The main menu's screen stack: owns the pages, loads their documents and
    // shows exactly one at a time. Back follows each page's declared parent,
    // so adding a screen is adding a page, nothing else.
    class Navigator final : public AbstractNavigator {
    public:
        // loads every page into the context; throws when one fails
        Navigator(Rml::Context &context, std::vector<std::unique_ptr<MenuPage>> pages, PageId root);

        void go(PageId page) override;
        void back() override;

        void show_root();
        void hide_all();

        // Escape / gamepad B on the current page: the page first, else back();
        // false when nothing happened (already on the root)
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
