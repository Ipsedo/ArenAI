//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_DATA_MODEL_H
#define ARENAI_DESKTOP_GUI_RML_DATA_MODEL_H

#include <functional>
#include <initializer_list>

#include <RmlUi/Core.h>

namespace arenai::desktop::gui {

    // One RmlUi data model: creation, bindings and dirty tracking behind a
    // single object, so a view registers its variables and callbacks in one
    // place and never juggles the constructor / handle pair.
    class DataModel {
    public:
        using EventHandler = std::function<void(const Rml::VariantList &arguments)>;

        // throws when RmlUi refuses the model (duplicate name)
        DataModel(Rml::Context &context, const Rml::String &name);

        template<typename T>
        void bind(const Rml::String &variable, T *value) {
            constructor_.Bind(variable, value);
        }

        template<typename T>
        Rml::StructHandle<T> register_struct() {
            return constructor_.RegisterStruct<T>();
        }

        template<typename T>
        void register_array() {
            constructor_.RegisterArray<T>();
        }

        void on_event(const Rml::String &name, EventHandler handler);

        void dirty(const Rml::String &variable);
        void dirty(std::initializer_list<const char *> variables);
        void dirty_all();

    private:
        // the model's own type register: each view declares the types it
        // binds (string arrays, row structs) without colliding with another
        // view's declarations; outlives the model, as RmlUi requires
        Rml::DataTypeRegister types_;
        Rml::DataModelConstructor constructor_;
        Rml::DataModelHandle handle_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_DATA_MODEL_H
