#ifndef ARENAI_DESKTOP_GUI_RML_DOCUMENT_VIEW_H
#define ARENAI_DESKTOP_GUI_RML_DOCUMENT_VIEW_H

#include <memory>

#include <RmlUi/Core.h>

#include "./data_model.h"

namespace arenai::desktop::gui {

    class DocumentView {
    public:
        virtual ~DocumentView() = default;
        DocumentView(const DocumentView &) = delete;
        DocumentView &operator=(const DocumentView &) = delete;

        void load(Rml::Context &context);

        void show(
            Rml::ModalFlag modal = Rml::ModalFlag::None,
            Rml::FocusFlag focus = Rml::FocusFlag::Auto);
        void hide();
        bool visible() const;

        Rml::ElementDocument &document() const;

        virtual void update() {}
        virtual void after_context_update() {}

    protected:
        DocumentView(const char *document_path, const char *model_name);

        virtual void bind(DataModel &) {}
        virtual void on_attached(Rml::ElementDocument &) {}
        virtual void on_show() {}
        virtual void on_hide() {}

        DataModel &model() const;
        Rml::Context &context() const;

    private:
        const char *document_path_;
        const char *model_name_;

        std::unique_ptr<DataModel> model_;
        Rml::ElementDocument *document_ = nullptr;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DOCUMENT_VIEW_H
