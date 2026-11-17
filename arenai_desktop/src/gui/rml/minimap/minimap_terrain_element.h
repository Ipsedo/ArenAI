#ifndef ARENAI_DESKTOP_GUI_RML_MINIMAP_MINIMAP_TERRAIN_ELEMENT_H
#define ARENAI_DESKTOP_GUI_RML_MINIMAP_MINIMAP_TERRAIN_ELEMENT_H

#include <memory>

#include <glm/glm.hpp>
#include <RmlUi/Core.h>

#include "../../gui.h"

namespace arenai::desktop::gui {

    // <minimap-terrain>: a window of the terrain texture stretched over the element's box
    class MinimapTerrainElement final : public Rml::Element {
    public:
        explicit MinimapTerrainElement(const Rml::String &tag);

        void set_terrain(const MinimapTerrain &terrain);
        // texture coordinates, may overflow [0, 1]
        void set_view(glm::vec2 view_min, glm::vec2 view_max);

        static constexpr const char *TAG = "minimap-terrain";

    protected:
        void OnRender() override;

    private:
        std::shared_ptr<const MinimapTerrain> terrain_;
        Rml::CallbackTexture texture_;
        Rml::Geometry geometry_;

        glm::vec2 view_min_;
        glm::vec2 view_max_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_MINIMAP_MINIMAP_TERRAIN_ELEMENT_H
