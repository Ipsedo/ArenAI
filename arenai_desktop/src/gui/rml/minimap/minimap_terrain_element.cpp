#include "./minimap_terrain_element.h"

#include <utility>

namespace arenai::desktop::gui {

    MinimapTerrainElement::MinimapTerrainElement(const Rml::String &tag)
        : Rml::Element(tag), view_min_(0.f), view_max_(1.f) {}

    void MinimapTerrainElement::set_terrain(const MinimapTerrain &terrain) {
        terrain_ = std::make_shared<const MinimapTerrain>(terrain);
        texture_ = {};
    }

    void MinimapTerrainElement::set_view(const glm::vec2 view_min, const glm::vec2 view_max) {
        view_min_ = view_min;
        view_max_ = view_max;
    }

    void MinimapTerrainElement::OnRender() {
        Rml::RenderManager *render_manager = GetRenderManager();
        if (!terrain_ || terrain_->width <= 0 || terrain_->height <= 0 || !render_manager) return;

        if (!texture_)
            texture_ = render_manager->MakeCallbackTexture(
                [terrain = terrain_](const Rml::CallbackTextureInterface &texture_interface) {
                    return texture_interface.GenerateTexture(
                        {reinterpret_cast<const Rml::byte *>(terrain->rgba.data()),
                         terrain->rgba.size()},
                        {terrain->width, terrain->height});
                });

        // the part of the window the texture covers, the rest keeps the background
        const glm::vec2 shown_min = glm::max(view_min_, glm::vec2(0.f));
        const glm::vec2 shown_max = glm::min(view_max_, glm::vec2(1.f));
        if (shown_min.x >= shown_max.x || shown_min.y >= shown_max.y) return;

        const Rml::Vector2f size = GetBox().GetSize(Rml::BoxArea::Border);
        const glm::vec2 box(size.x, size.y);
        const glm::vec2 span = view_max_ - view_min_;
        const glm::vec2 from = (shown_min - view_min_) / span * box;
        const glm::vec2 to = (shown_max - view_min_) / span * box;

        Rml::Mesh mesh;
        Rml::MeshUtilities::GenerateQuad(
            mesh, {from.x, from.y}, {to.x - from.x, to.y - from.y},
            Rml::ColourbPremultiplied(255, 255, 255, 255), {shown_min.x, shown_min.y},
            {shown_max.x, shown_max.y});
        geometry_ = render_manager->MakeGeometry(std::move(mesh));
        geometry_.Render(GetAbsoluteOffset(Rml::BoxArea::Border), texture_);
    }

}// namespace arenai::desktop::gui
