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
            {.path = "font/ChakraPetch-SemiBold.ttf", .family = "Chakra Petch", .weight = 600},
            {.path = "font/ChakraPetch-Bold.ttf", .family = "Chakra Petch", .weight = 700},
            {.path = "font/JetBrainsMono-Regular.ttf", .family = "JetBrains Mono", .weight = 400},
            {.path = "font/JetBrainsMono-Medium.ttf", .family = "JetBrains Mono", .weight = 500},
            {.path = "font/JetBrainsMono-SemiBold.ttf", .family = "JetBrains Mono", .weight = 600},
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
