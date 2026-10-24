//
// Created by samuel on 25/10/2026.
//

#include "./menu_page.h"

#include <stdexcept>

namespace arenai::desktop::gui {

    const char *to_string(const PageId page) {
        switch (page) {
            case PageId::Main: return "main";
            case PageId::Parameters: return "parameters";
            case PageId::Graphics: return "graphics";
            case PageId::Controls: return "controls";
            default: return "ai";
        }
    }

    std::optional<PageId> page_id_from_string(const std::string_view name) {
        for (const auto page:
             {PageId::Main, PageId::Parameters, PageId::Graphics, PageId::Controls, PageId::Ai})
            if (name == to_string(page)) return page;
        return std::nullopt;
    }

    MenuPage::MenuPage(
        const PageId id, const std::optional<PageId> parent, const char *document_path,
        const char *model_name)
        : DocumentView(document_path, model_name), id_(id), parent_(parent) {}

    void MenuPage::attach_navigator(AbstractNavigator &navigator) { navigator_ = &navigator; }

    AbstractNavigator &MenuPage::navigator() const {
        if (navigator_ == nullptr) throw std::logic_error("menu page used without a navigator");
        return *navigator_;
    }

    void MenuPage::bind(DataModel &model) {
        model.on_event("back", [this](const Rml::VariantList &) { navigator().back(); });
        model.on_event("navigate", [this](const Rml::VariantList &arguments) {
            if (arguments.empty()) return;
            if (const auto target = page_id_from_string(arguments[0].Get<Rml::String>()))
                navigator().go(*target);
        });
        bind_controls(model);
    }

}// namespace arenai::desktop::gui
