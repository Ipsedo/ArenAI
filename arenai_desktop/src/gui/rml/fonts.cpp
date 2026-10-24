//
// Created by samuel on 25/10/2026.
//

#include "./fonts.h"

#include <iterator>

#include <RmlUi/Core.h>

namespace arenai::desktop::gui {

    namespace {

        struct FontSpec {
            const char *path;
            const char *family;
            int weight;
        };

        constexpr FontSpec MENU_FONTS[] = {
            {.path = "font/Sora-Regular.ttf", .family = "Sora", .weight = 400},
            {.path = "font/Sora-SemiBold.ttf", .family = "Sora", .weight = 600},
            {.path = "font/Sora-Bold.ttf", .family = "Sora", .weight = 700},
            {.path = "font/IBMPlexMono-Regular.ttf", .family = "IBM Plex Mono", .weight = 400},
            {.path = "font/IBMPlexMono-Medium.ttf", .family = "IBM Plex Mono", .weight = 500},
            {.path = "font/IBMPlexMono-SemiBold.ttf", .family = "IBM Plex Mono", .weight = 600},
        };

    }// namespace

    std::vector<std::string>
    load_menu_fonts(const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader) {
        std::vector<std::string> buffers;
        buffers.reserve(std::size(MENU_FONTS));
        for (const auto &[path, family, weight]: MENU_FONTS) {
            buffers.push_back(asset_reader->read_text(path));
            const auto &buffer = buffers.back();
            Rml::LoadFontFace(
                Rml::Span(reinterpret_cast<const Rml::byte *>(buffer.data()), buffer.size()),
                family, Rml::Style::FontStyle::Normal, static_cast<Rml::Style::FontWeight>(weight));
        }
        return buffers;
    }

}// namespace arenai::desktop::gui
