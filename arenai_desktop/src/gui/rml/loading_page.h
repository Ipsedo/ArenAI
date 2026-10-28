//
// Created by samuel on 28/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_LOADING_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_LOADING_PAGE_H

#include "./document_view.h"

namespace arenai::desktop::gui {

    // the loading document shown while a match is being built: no data
    // model, no interactive element — all its motion is RCSS keyframes
    class LoadingPage final : public DocumentView {
    public:
        LoadingPage();
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_LOADING_PAGE_H
