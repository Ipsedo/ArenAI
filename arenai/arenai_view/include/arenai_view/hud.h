//
// Created by samuel on 26/03/2023.
//

#ifndef ARENAI_HUD_H
#define ARENAI_HUD_H

namespace arenai::view {

    class AbstractHudDrawable {
    public:
        virtual void draw(int width, int height) = 0;
        virtual ~AbstractHudDrawable() = default;
    };

}// namespace arenai::view

#endif// ARENAI_HUD_H
