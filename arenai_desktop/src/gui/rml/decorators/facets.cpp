//
// Created by samuel on 27/10/2026.
//

#include "./facets.h"

#include <algorithm>
#include <charconv>
#include <string_view>
#include <system_error>

namespace arenai::desktop::gui {

    namespace {

        // the mockups' artboard: every coordinate is a fraction of it
        constexpr float DESIGN_WIDTH = 1280.f;
        constexpr float DESIGN_HEIGHT = 720.f;

        Rml::ColourbPremultiplied scaled(const Rml::Colourb color, const float opacity) {
            return color.ToPremultiplied(std::clamp(opacity, 0.f, 1.f));
        }

        // convex (or single-reflex) polygon as a triangle fan from its first
        // vertex: enough for the hand-drawn facets of the backdrops
        void append_fan(
            Rml::Mesh &mesh, const std::vector<Rml::Vector2f> &points,
            const Rml::ColourbPremultiplied color) {
            const int base = static_cast<int>(mesh.vertices.size());
            for (const Rml::Vector2f position: points) {
                Rml::Vertex vertex;
                vertex.position = position;
                vertex.colour = color;
                vertex.tex_coord = {0.f, 0.f};
                mesh.vertices.push_back(vertex);
            }
            for (int i = 1; i + 1 < static_cast<int>(points.size()); ++i) {
                mesh.indices.push_back(base);
                mesh.indices.push_back(base + i);
                mesh.indices.push_back(base + i + 1);
            }
        }

    }// namespace

    FacetsDecorator::FacetsDecorator(
        const Rml::Colourb fill_color, const Rml::Colourb line_color,
        const Rml::NumericValue thickness, const Rml::String &polygons)
        : fill_color_(fill_color), line_color_(line_color), thickness_(thickness),
          polygons_(parse(polygons)) {}

    std::vector<FacetsDecorator::Polygon> FacetsDecorator::parse(const Rml::String &polygons) {
        // std::from_chars throughout: locale-independent, unlike strtof and
        // the stream extractors (the game may run under a locale whose
        // decimal separator is the comma, which would read "0.22" as 0)
        const auto read_float = [](std::string_view text, float &value) -> std::string_view {
            while (!text.empty() && text.front() == ' ') text.remove_prefix(1);
            const auto [end, error] =
                std::from_chars(text.data(), text.data() + text.size(), value);
            return error == std::errc() ? text.substr(static_cast<size_t>(end - text.data()))
                                        : std::string_view();
        };

        std::vector<Polygon> parsed;
        std::string_view rest(polygons);
        while (!rest.empty()) {
            const auto bar = rest.find('|');
            std::string_view entry = rest.substr(0, bar);
            rest = bar == std::string_view::npos ? std::string_view() : rest.substr(bar + 1);

            Polygon polygon;
            if (const auto at = entry.find('@'); at != std::string_view::npos) {
                read_float(entry.substr(at + 1), polygon.opacity);
                entry = entry.substr(0, at);
            }
            for (float x = 0.f, y = 0.f;;) {
                const std::string_view after_x = read_float(entry, x);
                if (after_x.data() == nullptr) break;
                const std::string_view after_y = read_float(after_x, y);
                if (after_y.data() == nullptr) break;
                polygon.points.emplace_back(x, y);
                entry = after_y;
            }
            if (polygon.points.size() >= 2) parsed.push_back(std::move(polygon));
        }
        return parsed;
    }

    Rml::Mesh FacetsDecorator::build_mesh(Rml::Element &element) const {
        const float opacity = element.GetComputedValues().opacity();
        const Rml::Vector2f size = element_size(element);
        const Rml::Vector2f scale(size.x / DESIGN_WIDTH, size.y / DESIGN_HEIGHT);
        const float thickness = element.ResolveLength(thickness_);
        const bool fills = fill_color_.alpha > 0;
        const bool lines = line_color_.alpha > 0 && thickness > 0.f;

        Rml::Mesh mesh;
        std::vector<Rml::Vector2f> points;
        for (const auto &polygon: polygons_) {
            points.clear();
            for (const Rml::Vector2f point: polygon.points)
                points.emplace_back(point.x * scale.x, point.y * scale.y);
            const float alpha = opacity * polygon.opacity;

            if (points.size() == 2) {
                // a bare segment: drawn with whichever color is set, the line
                // color first
                const Rml::Colourb color = lines ? line_color_ : fill_color_;
                if (lines || fills)
                    append_bar(mesh, points[0], points[1], thickness, scaled(color, alpha));
                continue;
            }

            if (fills) append_fan(mesh, points, scaled(fill_color_, alpha));
            if (lines) {
                const Rml::ColourbPremultiplied line = scaled(line_color_, alpha);
                for (std::size_t i = 0; i < points.size(); ++i)
                    append_bar(mesh, points[i], points[(i + 1) % points.size()], thickness, line);
            }
        }
        return mesh;
    }

    FacetsDecoratorInstancer::FacetsDecoratorInstancer()
        : id_fill_color_(add_color("fill-color", "transparent")),
          id_line_color_(add_color("line-color", "transparent")),
          id_thickness_(add_length("thickness", "1dp")), id_polygons_(add_string("polygons", "")) {
        register_shorthand();
    }

    Rml::SharedPtr<Rml::Decorator> FacetsDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<FacetsDecorator>(
            color_of(properties, id_fill_color_), color_of(properties, id_line_color_),
            length_of(properties, id_thickness_), string_of(properties, id_polygons_));
    }

}// namespace arenai::desktop::gui
