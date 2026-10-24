//
// Created by samuel on 25/10/2026.
//

#include "./document_view.h"

#include <stdexcept>
#include <string>

namespace arenai::desktop::gui {

    DocumentView::DocumentView(const char *document_path, const char *model_name)
        : document_path_(document_path), model_name_(model_name) {}

    void DocumentView::load(Rml::Context &context) {
        if (model_name_ != nullptr) {
            model_ = std::make_unique<DataModel>(context, model_name_);
            bind(*model_);
        }

        document_ = context.LoadDocument(document_path_);
        if (document_ == nullptr)
            throw std::runtime_error(
                std::string("RmlUi document failed to load: ") + document_path_);

        on_attached(*document_);
    }

    void DocumentView::show(const Rml::ModalFlag modal, const Rml::FocusFlag focus) {
        if (model_) model_->dirty_all();
        document_->Show(modal, focus);
        on_show();
    }

    void DocumentView::hide() {
        if (!document_->IsVisible()) return;
        on_hide();
        document_->Hide();
    }

    bool DocumentView::visible() const { return document_ != nullptr && document_->IsVisible(); }

    Rml::ElementDocument &DocumentView::document() const { return *document_; }

    void DocumentView::set_class(const Rml::String &class_name, const bool activate) const {
        document_->SetClass(class_name, activate);
    }

    DataModel &DocumentView::model() const { return *model_; }

    Rml::Context &DocumentView::context() const { return *document_->GetContext(); }

}// namespace arenai::desktop::gui
