#ifndef ARENAI_DESKTOP_GUI_RML_MINIMAP_MINIMAP_WIDGET_H
#define ARENAI_DESKTOP_GUI_RML_MINIMAP_MINIMAP_WIDGET_H

#include <cstddef>
#include <optional>
#include <vector>

#include <glm/glm.hpp>
#include <RmlUi/Core.h>

#include "../../gui.h"
#include "./minimap_terrain_element.h"

namespace arenai::desktop::gui {

    // the #minimap block of hud.rml: terrain, player cursor and enemy blips
    class MinimapWidget final : public AbstractMinimap {
    public:
        // throws when the document misses one of the minimap elements
        explicit MinimapWidget(Rml::ElementDocument &document);

        void set_terrain(const MinimapTerrain &terrain) override;
        void update(const MinimapSnapshot &snapshot) override;

    private:
        // grows the pool on demand
        Rml::Element &blip(std::size_t index);
        void show_coordinates(glm::ivec2 coordinates);

        Rml::ElementDocument &document_;
        MinimapTerrainElement *terrain_ = nullptr;
        Rml::Element *blips_ = nullptr;
        Rml::Element *cursor_ = nullptr;
        Rml::Element *cone_ = nullptr;
        Rml::Element *coordinates_ = nullptr;

        std::vector<Rml::Element *> blip_pool_;
        std::optional<glm::ivec2> shown_coordinates_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_MINIMAP_MINIMAP_WIDGET_H
