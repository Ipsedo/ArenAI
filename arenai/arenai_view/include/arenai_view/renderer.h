//
// Created by samuel on 18/03/2023.
//

#ifndef ARENAI_RENDERER_H
#define ARENAI_RENDERER_H

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <glm/glm.hpp>

#include "./drawable.h"
#include "./hud.h"

namespace arenai::view {

    template<typename T>
    struct image {
        std::vector<T> pixels;
    };

    class AbstractRenderer {
    public:
        virtual ~AbstractRenderer() = default;

        virtual void
        add_drawable(const std::string &name, std::unique_ptr<AbstractDrawable> drawable) = 0;
        virtual void remove_drawable(const std::string &name) = 0;

        virtual void
        draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) = 0;

        virtual int get_width() const = 0;
        virtual int get_height() const = 0;

        virtual void make_current() const = 0;
        virtual void release_current() const = 0;
    };

    class AbstractPlayerRenderer : public virtual AbstractRenderer {
    public:
        virtual void add_hud_drawable(std::unique_ptr<AbstractHudDrawable> hud_drawable) = 0;
        virtual void set_window_size(int new_width, int new_height) = 0;
    };

    class AbstractOffscreenRenderer : public virtual AbstractRenderer {
    public:
        virtual image<uint8_t> draw_and_get_frame(
            const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) = 0;
    };

}// namespace arenai::view

#endif// ARENAI_RENDERER_H
