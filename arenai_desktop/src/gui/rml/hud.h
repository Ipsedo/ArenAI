//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_HUD_H
#define ARENAI_DESKTOP_GUI_RML_HUD_H

#include <cstddef>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

#include "../gui.h"
#include "./document_view.h"

namespace arenai::desktop::gui {

    // what draws a UI pass over the frame already rendered
    class AbstractOverlayRenderer {
    public:
        virtual ~AbstractOverlayRenderer() = default;
        virtual void render_overlay() = 0;
    };

    // The in-game overlay of hud.rml: the aim reticle, the Battlefield-style
    // hit marker and the incoming-damage arc pool.
    class HudOverlay final : public DocumentView, public AbstractHudOverlay {
    public:
        explicit HudOverlay(AbstractOverlayRenderer &renderer);

        void notify_hit(HitKind kind) override;
        void notify_damage(float screen_angle) override;
        void set_aim_point(std::optional<glm::vec2> normalized) override;
        void render() override;

    protected:
        // grabs the overlay elements; throws when hud.rml misses one
        void on_attached(Rml::ElementDocument &document) override;

    private:
        AbstractOverlayRenderer &renderer_;

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
