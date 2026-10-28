//
// Created by samuel on 28/10/2026.
//

#include "./crystals.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <string_view>
#include <system_error>

namespace arenai::desktop::gui {

    namespace {

        // the mockups' artboard: every coordinate is a fraction of it
        constexpr float DESIGN_WIDTH = 1280.f;
        constexpr float DESIGN_HEIGHT = 720.f;

        struct Vec3 {
            float x, y, z;
        };

        // the two authored irregular tetrahedra, in unit crystal space
        constexpr std::array<Vec3, 4> SHAPE_1{
            {{0.f, -1.7f, 0.15f},
             {1.05f, 0.6f, 0.6f},
             {-0.8f, 0.75f, 0.55f},
             {-0.15f, 0.5f, -1.05f}}};
        constexpr std::array<Vec3, 4> SHAPE_2{
            {{0.f, -1.35f, -0.3f},
             {0.95f, 0.8f, 0.4f},
             {-1.15f, 0.45f, 0.3f},
             {0.2f, 0.35f, 1.05f}}};
        constexpr std::array<std::array<int, 2>, 6> EDGES{
            {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};

    }// namespace

    CrystalsDecorator::CrystalsDecorator(
        const Rml::Colourb line_color, const Rml::NumericValue thickness,
        const Rml::String &instances)
        : line_color_(line_color), thickness_(thickness), instances_(parse(instances)) {}

    std::vector<CrystalsDecorator::Instance>
    CrystalsDecorator::parse(const Rml::String &instances) {
        // std::from_chars: locale-independent, like the facets parser
        const auto read_float = [](std::string_view text, float &value) -> std::string_view {
            while (!text.empty() && text.front() == ' ') text.remove_prefix(1);
            const auto [end, error] =
                std::from_chars(text.data(), text.data() + text.size(), value);
            return error == std::errc() ? text.substr(static_cast<size_t>(end - text.data()))
                                        : std::string_view();
        };

        std::vector<Instance> parsed;
        std::string_view rest(instances);
        while (!rest.empty()) {
            const auto bar = rest.find('|');
            std::string_view entry = rest.substr(0, bar);
            rest = bar == std::string_view::npos ? std::string_view() : rest.substr(bar + 1);

            std::array<float, 6> fields{};
            bool complete = true;
            for (float &field: fields) {
                entry = read_float(entry, field);
                if (entry.data() == nullptr) {
                    complete = false;
                    break;
                }
            }
            if (!complete) continue;

            parsed.push_back(Instance{
                .center = {fields[0], fields[1]},
                .size = fields[2],
                .shape = fields[3] >= 1.5f ? 2 : 1,
                .spin = fields[4],
                .phase = fields[5]});
        }
        return parsed;
    }

    Rml::Mesh CrystalsDecorator::build_mesh(Rml::Element &element) const {
        const float opacity = element.GetComputedValues().opacity();
        const Rml::Vector2f size = element_size(element);
        const Rml::Vector2f scale(size.x / DESIGN_WIDTH, size.y / DESIGN_HEIGHT);
        const float thickness = element.ResolveLength(thickness_);
        const auto t = static_cast<float>(Rml::GetSystemInterface()->GetElapsedTime());

        Rml::Mesh mesh;
        for (const Instance &instance: instances_) {
            const auto &shape = instance.shape == 2 ? SHAPE_2 : SHAPE_1;
            const float ax = t * 0.3f * instance.spin + instance.phase;
            const float ay = t * 0.22f * instance.spin + 2.f * instance.phase;
            const float sin_x = std::sin(ax), cos_x = std::cos(ax);
            const float sin_y = std::sin(ay), cos_y = std::cos(ay);
            const float bob = std::sin(t * 0.4f + instance.phase) * 7.f;

            std::array<Rml::Vector2f, 4> projected;
            std::array<float, 4> depth{};
            for (std::size_t i = 0; i < shape.size(); ++i) {
                const Vec3 v = shape[i];
                const float y1 = v.y * cos_x - v.z * sin_x;
                const float z1 = v.y * sin_x + v.z * cos_x;
                const float x2 = v.x * cos_y + z1 * sin_y;
                const float z2 = -v.x * sin_y + z1 * cos_y;
                // light perspective: the far half of the crystal shrinks
                const float k = 1.f / (1.f + z2 * 0.17f);
                projected[i] = {
                    (instance.center.x + x2 * instance.size * k) * scale.x,
                    (instance.center.y + y1 * instance.size * k + bob) * scale.y};
                depth[i] = z2;
            }

            for (const auto &[from, to]: EDGES) {
                // back edges fade: the wireframe reads as a volume
                const float z = (depth[from] + depth[to]) * 0.5f;
                const float alpha =
                    opacity * std::clamp(0.16f + 0.24f * (0.5f - z / 3.f), 0.f, 1.f);
                append_bar(
                    mesh, projected[from], projected[to], thickness,
                    line_color_.ToPremultiplied(alpha));
            }
        }
        return mesh;
    }

    CrystalsDecoratorInstancer::CrystalsDecoratorInstancer()
        : id_line_color_(add_color("line-color", "transparent")),
          id_thickness_(add_length("thickness", "1dp")),
          id_instances_(add_string("instances", "")) {
        register_shorthand();
    }

    Rml::SharedPtr<Rml::Decorator> CrystalsDecoratorInstancer::InstanceDecorator(
        const Rml::String &, const Rml::PropertyDictionary &properties,
        const Rml::DecoratorInstancerInterface &) {
        return Rml::MakeShared<CrystalsDecorator>(
            color_of(properties, id_line_color_), length_of(properties, id_thickness_),
            string_of(properties, id_instances_));
    }

}// namespace arenai::desktop::gui
