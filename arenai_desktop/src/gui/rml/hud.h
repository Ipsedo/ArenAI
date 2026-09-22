//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_HUD_H
#define ARENAI_DESKTOP_GUI_RML_HUD_H

#include <cstddef>
#include <optional>
#include <vector>

#include <glm/glm.hpp>
#include <RmlUi/Core.h>

#include "../menu.h"

namespace arenai::desktop::gui {

    // The in-game overlay elements of hud.rml: the aim reticle, the
    // Battlefield-style hit marker and the incoming-damage arc pool.
    class Hud {
    public:
        // grabs the overlay elements from the loaded hud document
        explicit Hud(Rml::ElementDocument &document);

        void notify_hit(HitKind kind) const;
        void notify_damage(float screen_angle);
        void set_aim_point(std::optional<glm::vec2> normalized, int width, int height) const;

    private:
        Rml::Element *reticle_ = nullptr;
        Rml::Element *hit_marker_ = nullptr;
        std::vector<Rml::Element *> damage_arcs_;
        std::size_t next_damage_arc_ = 0;

        static constexpr float DAMAGE_ARC_FADE_SECONDS = 0.8f;
        static constexpr float HIT_MARKER_FADE_SECONDS = 0.45f;
        static constexpr float HIT_MARKER_END_SCALE = 1.3f;
        static constexpr float KILL_MARKER_FADE_SECONDS = 0.6f;
        static constexpr float KILL_MARKER_START_SCALE = 1.1f;
        static constexpr float KILL_MARKER_END_SCALE = 1.55f;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_HUD_H
