//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_DOCUMENT_VIEW_H
#define ARENAI_DESKTOP_GUI_RML_DOCUMENT_VIEW_H

#include <memory>

#include <RmlUi/Core.h>

#include "./data_model.h"

namespace arenai::desktop::gui {

    // One .rml document with its own data model (Template Method). load()
    // runs the fixed RmlUi sequence — create the model, let the subclass
    // bind its variables and callbacks, load the document, let the subclass
    // grab its elements — so a view only writes the hooks. Every data model
    // of a view is dirtied on show(): views are shown one at a time and may
    // mirror state another view changed meanwhile.
    class DocumentView {
    public:
        virtual ~DocumentView() = default;
        DocumentView(const DocumentView &) = delete;
        DocumentView &operator=(const DocumentView &) = delete;

        // throws when the document fails to load
        void load(Rml::Context &context);

        void show(
            Rml::ModalFlag modal = Rml::ModalFlag::None,
            Rml::FocusFlag focus = Rml::FocusFlag::Auto);
        void hide();
        bool visible() const;

        Rml::ElementDocument &document() const;
        void set_class(const Rml::String &class_name, bool activate) const;

        // once per frame while visible, before the context update
        virtual void update() {}
        // once per frame while visible, after the context update rebuilt the
        // data-bound clones
        virtual void after_context_update() {}

    protected:
        // model_name nullptr = the document binds no data
        DocumentView(const char *document_path, const char *model_name);

        virtual void bind(DataModel &) {}
        virtual void on_attached(Rml::ElementDocument &) {}
        virtual void on_show() {}
        virtual void on_hide() {}

        // valid once loaded, for views declaring a model
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
