//
// Created by samuel on 28/10/2026.
//

#include "./text_gradient.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/ElementText.h>
#include <RmlUi/Core/FontEngineInterface.h>
#include <RmlUi/Core/RenderManager.h>
#include <RmlUi/Core/StringUtilities.h>
#include <RmlUi/Core/TextShapingContext.h>

namespace arenai::desktop::gui {

    namespace {

        // one compiled glyph run per font texture page
        struct TextGradientData {
            struct Run {
                Rml::Geometry geometry;
                Rml::Texture texture;
            };
            std::vector<Run> runs;
        };

        // the element's direct text nodes concatenated (the title is one),
        // cased like the hidden native text (capitalize is not supported)
        Rml::String element_text(Rml::Element &element) {
            Rml::String text;
            for (int i = 0; i < element.GetNumChildren(); ++i)
                if (const auto *child = rmlui_dynamic_cast<Rml::ElementText *>(element.GetChild(i)))
                    text += child->GetText();
            switch (element.GetComputedValues().text_transform()) {
                case Rml::Style::TextTransform::Uppercase:
                    return Rml::StringUtilities::ToUpper(std::move(text));
                case Rml::Style::TextTransform::Lowercase:
                    return Rml::StringUtilities::ToLower(std::move(text));
                default: return text;
            }
        }

        Rml::ColourbPremultiplied blend(
            const Rml::Colourb left, const Rml::Colourb right, const float t, const float opacity) {
            const auto mix = [t](const Rml::byte a, const Rml::byte b) {
                return static_cast<Rml::byte>(std::lround(
                    static_cast<float>(a) + (static_cast<float>(b) - static_cast<float>(a)) * t));
            };
            return Rml::Colourb(
                       mix(left.red, right.red), mix(left.green, right.green),
                       mix(left.blue, right.blue), mix(left.alpha, right.alpha))
                .ToPremultiplied(opacity);
        }

    }// namespace

    TextGradientDecorator::TextGradientDecorator(
        const Rml::Colourb left_color, const Rml::Colourb right_color, const float start_percent,
        const float end_percent)
        : left_color_(left_color), right_color_(right_color), start_percent_(start_percent),
          end_percent_(end_percent) {}

    Rml::DecoratorDataHandle
    TextGradientDecorator::GenerateElementData(Rml::Element *element, Rml::BoxArea) const {
        auto *data = new TextGradientData();

        Rml::RenderManager *render_manager = element->GetRenderManager();
        const Rml::FontFaceHandle face = element->GetFontFaceHandle();
        const Rml::String text = element_text(*element);
        if (render_manager == nullptr || face == 0 || text.empty())
            return reinterpret_cast<Rml::DecoratorDataHandle>(data);

        const auto &computed = element->GetComputedValues();
        const Rml::TextShapingContext shaping{
            computed.language(), computed.direction(), computed.font_kerning(),
            computed.letter_spacing()};
        const float opacity = computed.opacity();

        // the single line sits where the hidden native text does: baseline at
        // the content box top plus the font's ascent
        const Rml::Vector2f origin = element->GetBox().GetPosition(Rml::BoxArea::Content);
        const float ascent = Rml::GetFontEngineInterface()->GetFontMetrics(face).ascent;

        Rml::TexturedMeshList meshes;
        const int width = Rml::GetFontEngineInterface()->GenerateString(
            *render_manager, face, 0, text, origin + Rml::Vector2f(0.f, ascent),
            Rml::ColourbPremultiplied(255, 255, 255, 255), opacity, shaping, meshes);

        const float x_start = origin.x + static_cast<float>(width) * start_percent_ * 0.01f;
        const float x_end = origin.x + static_cast<float>(width) * end_percent_ * 0.01f;
        const float span = x_end - x_start;
        for (auto &[mesh, texture]: meshes) {
            for (auto &vertex: mesh.vertices) {
                const float t = span > 0.f
                                    ? std::clamp((vertex.position.x - x_start) / span, 0.f, 1.f)
                                    : (vertex.position.x < x_start ? 0.f : 1.f);
                vertex.colour = blend(left_color_, right_color_, t, opacity);
            }
            data->runs.push_back({render_manager->MakeGeometry(std::move(mesh)), texture});
        }
        return reinterpret_cast<Rml::DecoratorDataHandle>(data);
    }

    void
    TextGradientDecorator::ReleaseElementData(const Rml::DecoratorDataHandle element_data) const {
        delete reinterpret_cast<TextGradientData *>(element_data);
    }

    void TextGradientDecorator::RenderElement(
        Rml::Element *element, const Rml::DecoratorDataHandle element_data) const {
        const Rml::Vector2f translation = element->GetAbsoluteOffset(Rml::BoxArea::Border);
        for (auto &run: reinterpret_cast<TextGradientData *>(element_data)->runs)
            run.geometry.Render(translation, run.texture);
    }

    TextGradientDecoratorInstancer::TextGradientDecoratorInstancer()
        : id_left_color_(add_color("left-color", "white")),
          id_right_color_(add_color("right-color", "white")), id_start_(add_number("start", "36")),
          id_end_(add_number("end", "64")) {
        register_shorthand();
    }

    Rml::SharedPtr<Rml::Decorator> TextGradientDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<TextGradientDecorator>(
            color_of(properties, id_left_color_), color_of(properties, id_right_color_),
            number_of(properties, id_start_), number_of(properties, id_end_));
    }

}// namespace arenai::desktop::gui
