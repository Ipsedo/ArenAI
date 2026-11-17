#include "./minimap_widget.h"

#include <numbers>
#include <stdexcept>
#include <string>

namespace arenai::desktop::gui {

    namespace {

        Rml::Element &require(Rml::ElementDocument &document, const char *id) {
            Rml::Element *element = document.GetElementById(id);
            if (element == nullptr)
                throw std::runtime_error(std::string("hud.rml misses its #") + id + " element");
            return *element;
        }

        void rotate(Rml::Element &element, const float radians) {
            constexpr float rad_to_deg = 180.f / std::numbers::pi_v<float>;
            element.SetProperty(
                Rml::PropertyId::Transform,
                Rml::Transform::MakeProperty({Rml::Transforms::Rotate2D(radians * rad_to_deg)}));
        }

    }// namespace

    MinimapWidget::MinimapWidget(Rml::ElementDocument &document) : document_(document) {
        terrain_ = dynamic_cast<MinimapTerrainElement *>(&require(document, "minimap-terrain"));
        if (terrain_ == nullptr)
            throw std::runtime_error("#minimap-terrain is not a <minimap-terrain> element");
        blips_ = &require(document, "minimap-blips");
        cursor_ = &require(document, "minimap-cursor");
        cone_ = &require(document, "minimap-cone");
        coordinates_ = &require(document, "minimap-coordinates");
    }

    void MinimapWidget::set_terrain(const MinimapTerrain &terrain) {
        terrain_->set_terrain(terrain);
    }

    void MinimapWidget::update(const MinimapSnapshot &snapshot) {
        terrain_->set_view(snapshot.view_min, snapshot.view_max);
        rotate(*cursor_, snapshot.player_heading);
        rotate(*cone_, snapshot.player_heading);
        show_coordinates(snapshot.player_coordinates);

        for (std::size_t i = 0; i < snapshot.blips.size(); i++) {
            const auto &[position, kind] = snapshot.blips[i];
            Rml::Element &element = blip(i);
            element.SetClass("dead", kind == BlipKind::DeadEnemy);
            element.SetClass("edge", kind == BlipKind::EnemyOutOfRange);
            element.SetProperty(
                Rml::PropertyId::Left, Rml::Property(position.x * 100.f, Rml::Unit::PERCENT));
            element.SetProperty(
                Rml::PropertyId::Top, Rml::Property(position.y * 100.f, Rml::Unit::PERCENT));
            element.SetProperty(
                Rml::PropertyId::Visibility, Rml::Property(Rml::Style::Visibility::Visible));
        }
        for (std::size_t i = snapshot.blips.size(); i < blip_pool_.size(); i++)
            blip_pool_[i]->SetProperty(
                Rml::PropertyId::Visibility, Rml::Property(Rml::Style::Visibility::Hidden));
    }

    Rml::Element &MinimapWidget::blip(const std::size_t index) {
        while (blip_pool_.size() <= index) {
            Rml::ElementPtr element = document_.CreateElement("div");
            element->SetClass("minimap-blip", true);
            blip_pool_.push_back(blips_->AppendChild(std::move(element)));
        }
        return *blip_pool_[index];
    }

    void MinimapWidget::show_coordinates(const glm::ivec2 coordinates) {
        if (shown_coordinates_ == coordinates) return;
        shown_coordinates_ = coordinates;
        coordinates_->SetInnerRML(
            "X " + std::to_string(coordinates.x) + "  Z " + std::to_string(coordinates.y));
    }

}// namespace arenai::desktop::gui
