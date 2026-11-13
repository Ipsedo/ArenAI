#ifndef ARENAI_DESKTOP_GUI_RML_FONTS_H
#define ARENAI_DESKTOP_GUI_RML_FONTS_H

#include <memory>
#include <string>
#include <vector>

#include <arenai_utils/file_reader.h>

namespace arenai::desktop::gui {

    std::vector<std::string>
    load_menu_fonts(const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader);

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_FONTS_H
