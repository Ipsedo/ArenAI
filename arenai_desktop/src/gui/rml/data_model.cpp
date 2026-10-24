//
// Created by samuel on 25/10/2026.
//

#include "./data_model.h"

#include <stdexcept>
#include <utility>

namespace arenai::desktop::gui {

    DataModel::DataModel(Rml::Context &context, const Rml::String &name)
        : constructor_(context.CreateDataModel(name, &types_)) {
        if (!constructor_) throw std::runtime_error("RmlUi data model creation failed: " + name);
        handle_ = constructor_.GetModelHandle();
    }

    void DataModel::on_event(const Rml::String &name, EventHandler handler) {
        constructor_.BindEventCallback(
            name, [handler = std::move(handler)](
                      Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                handler(arguments);
            });
    }

    void DataModel::dirty(const Rml::String &variable) { handle_.DirtyVariable(variable); }

    void DataModel::dirty(const std::initializer_list<const char *> variables) {
        for (const char *variable: variables) handle_.DirtyVariable(variable);
    }

    void DataModel::dirty_all() { handle_.DirtyAllVariables(); }

}// namespace arenai::desktop::gui
