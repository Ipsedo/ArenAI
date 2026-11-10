#ifndef ARENAI_DESKTOP_GUI_RML_MENU_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_MENU_PAGE_H

#include <optional>
#include <string_view>

#include "./document_view.h"

namespace arenai::desktop::gui {

    enum class PageId { Main, Parameters, Graphics, Controls, Ai };

    // canonical names, used by the RML `navigate('<name>')` event
    const char *to_string(PageId page);
    std::optional<PageId> page_id_from_string(std::string_view name);

    // what a page may ask of the screen stack
    class AbstractNavigator {
    public:
        virtual ~AbstractNavigator() = default;

        virtual void go(PageId page) = 0;
        // to the current page's parent; no-op on the root
        virtual void back() = 0;
    };

    class MenuPage : public DocumentView {
    public:
        PageId id() const { return id_; }
        std::optional<PageId> parent() const { return parent_; }

        // set by the Navigator before load()
        void attach_navigator(AbstractNavigator &navigator);

        virtual bool on_back() { return false; }

    protected:
        MenuPage(
            PageId id, std::optional<PageId> parent, const char *document_path,
            const char *model_name);

        // the page's own variables and callbacks
        virtual void bind_controls(DataModel &) {}

        AbstractNavigator &navigator() const;

    private:
        void bind(DataModel &model) final;

        PageId id_;
        std::optional<PageId> parent_;
        AbstractNavigator *navigator_ = nullptr;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_MENU_PAGE_H
