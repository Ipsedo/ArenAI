#ifndef ARENAI_DESKTOP_GUI_RML_MENU_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_MENU_PAGE_H

#include <optional>
#include <string_view>

#include "./document_view.h"

namespace arenai::desktop::gui {

    enum class PageId { Main, Parameters, Graphics, Controls, Ai };

    const char *to_string(PageId page);
    std::optional<PageId> page_id_from_string(std::string_view name);

    class AbstractNavigator {
    public:
        virtual ~AbstractNavigator() = default;

        virtual void go(PageId page) = 0;
        virtual void back() = 0;
    };

    class MenuPage : public DocumentView {
    public:
        PageId id() const { return id_; }
        std::optional<PageId> parent() const { return parent_; }

        void attach_navigator(AbstractNavigator &navigator);

        virtual bool on_back() { return false; }

    protected:
        MenuPage(
            PageId id, std::optional<PageId> parent, const char *document_path,
            const char *model_name);

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
