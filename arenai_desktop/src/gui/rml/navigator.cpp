//
// Created by samuel on 25/10/2026.
//

#include "./navigator.h"

#include <stdexcept>
#include <string>

namespace arenai::desktop::gui {

    Navigator::Navigator(
        Rml::Context &context, std::vector<std::unique_ptr<MenuPage>> pages, const PageId root)
        : root_(root) {
        for (auto &page: pages) {
            page->attach_navigator(*this);
            page->load(context);
            const auto id = page->id();
            if (!pages_.emplace(id, std::move(page)).second)
                throw std::logic_error(std::string("menu page registered twice: ") + to_string(id));
        }
        static_cast<void>(page(root_));// the root must exist
    }

    void Navigator::go(const PageId target) {
        MenuPage &next = page(target);
        if (current_ == &next) return;
        if (current_ != nullptr) current_->hide();
        current_ = &next;
        current_->show();
    }

    void Navigator::back() {
        if (current_ == nullptr) return;
        if (const auto parent = current_->parent()) go(*parent);
    }

    void Navigator::show_root() { go(root_); }

    void Navigator::hide_all() {
        for (const auto &[id, page]: pages_) page->hide();
        current_ = nullptr;
    }

    bool Navigator::on_escape() {
        if (current_ == nullptr) return false;
        if (current_->on_back()) return true;
        if (!current_->parent()) return false;
        back();
        return true;
    }

    void Navigator::update() const {
        if (current_ != nullptr) current_->update();
    }

    void Navigator::after_context_update() const {
        if (current_ != nullptr) current_->after_context_update();
    }

    void Navigator::set_gamepad_nav(const bool gamepad) const {
        for (const auto &[id, page]: pages_) page->set_class("gamepad-nav", gamepad);
    }

    MenuPage &Navigator::page(const PageId id) const {
        const auto found = pages_.find(id);
        if (found == pages_.end())
            throw std::logic_error(std::string("unknown menu page: ") + to_string(id));
        return *found->second;
    }

}// namespace arenai::desktop::gui
