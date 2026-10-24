//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_FONTS_H
#define ARENAI_DESKTOP_GUI_RML_FONTS_H

#include <memory>
#include <string>
#include <vector>

#include <arenai_utils/file_reader.h>

namespace arenai::desktop::gui {

    // The menu's font faces, registered with RmlUi under an explicit
    // family/weight (the static TTFs carry per-weight legacy family names
    // that would not match the RCSS font-family otherwise). The returned
    // buffers back the faces and must outlive Rml::Shutdown().
    std::vector<std::string>
    load_menu_fonts(const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader);

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_FONTS_H
