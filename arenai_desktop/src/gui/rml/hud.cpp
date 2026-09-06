//
// Created by samuel on 06/09/2026.
//

#include "./hud.h"

#include <numbers>
#include <stdexcept>

namespace arenai::desktop::gui {

    Hud::Hud(Rml::ElementDocument &document) {
        reticle_ = document.GetElementById("reticle");
        hit_marker_ = document.GetElementById("hit-marker");
        if (reticle_ == nullptr || hit_marker_ == nullptr)
            throw std::runtime_error("hud.rml misses its #reticle / #hit-marker elements");

        if (Rml::Element *arcs = document.GetElementById("damage-arcs"))
            for (int i = 0; i < arcs->GetNumChildren(); i++)
                damage_arcs_.push_back(arcs->GetChild(i));
        if (damage_arcs_.empty())
            throw std::runtime_error("hud.rml misses its #damage-arcs elements");
    }

    void Hud::notify_hit(const HitKind kind) const {
        const bool kill = kind == HitKind::Kill;
        hit_marker_->SetClass("kill", kill);

        // Battlefield-style feedback: the ticks spread outward while
        // fading; a kill starts bigger, flares wider and lasts longer
        const float duration = kill ? KILL_MARKER_FADE_SECONDS : HIT_MARKER_FADE_SECONDS;
        const Rml::Tween tween(Rml::Tween::Quadratic, Rml::Tween::Out);

        const Rml::Property opaque(1.f, Rml::Unit::NUMBER);
        hit_marker_->Animate(
            "opacity", Rml::Property(0.f, Rml::Unit::NUMBER), duration, tween, 1, false, 0.f,
            &opaque);

        const Rml::Property start_scale = Rml::Transform::MakeProperty(
            {Rml::Transforms::Scale2D(kill ? KILL_MARKER_START_SCALE : 1.f)});
        hit_marker_->Animate(
            "transform",
            Rml::Transform::MakeProperty(
                {Rml::Transforms::Scale2D(kill ? KILL_MARKER_END_SCALE : HIT_MARKER_END_SCALE)}),
            duration, tween, 1, false, 0.f, &start_scale);
    }

    void Hud::notify_damage(const float screen_angle) {
        // oldest-slot reuse: a burst of impacts shows as many arcs as
        // the pool holds, each rotated toward its own shooter
        Rml::Element *arc = damage_arcs_[next_damage_arc_];
        next_damage_arc_ = (next_damage_arc_ + 1) % damage_arcs_.size();

        constexpr float rad_to_deg = 180.f / std::numbers::pi_v<float>;
        arc->SetProperty(
            Rml::PropertyId::Transform,
            Rml::Transform::MakeProperty({Rml::Transforms::Rotate2D(screen_angle * rad_to_deg)}));

        const Rml::Tween tween(Rml::Tween::Quadratic, Rml::Tween::Out);
        const Rml::Property opaque(1.f, Rml::Unit::NUMBER);
        arc->Animate(
            "opacity", Rml::Property(0.f, Rml::Unit::NUMBER), DAMAGE_ARC_FADE_SECONDS, tween, 1,
            false, 0.f, &opaque);
    }

    void Hud::set_aim_point(
        const std::optional<glm::vec2> normalized, const int width, const int height) const {
        if (!normalized) {
            reticle_->SetProperty(
                Rml::PropertyId::Visibility, Rml::Property(Rml::Style::Visibility::Hidden));
            return;
        }
        reticle_->SetProperty(
            Rml::PropertyId::Visibility, Rml::Property(Rml::Style::Visibility::Visible));
        reticle_->SetProperty(
            Rml::PropertyId::Left,
            Rml::Property(normalized->x * static_cast<float>(width), Rml::Unit::PX));
        reticle_->SetProperty(
            Rml::PropertyId::Top,
            Rml::Property(normalized->y * static_cast<float>(height), Rml::Unit::PX));
    }

}// namespace arenai::desktop::gui
