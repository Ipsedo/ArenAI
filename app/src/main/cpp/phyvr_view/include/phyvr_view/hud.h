//
// Created by samuel on 26/03/2023.
//

#ifndef PHYVR_HUD_H
#define PHYVR_HUD_H

#include <functional>
#include <memory>

#include <phyvr_controller/inputs.h>
#include <phyvr_utils/file_reader.h>

#include "./program.h"

class HUDDrawable {
public:
    virtual void draw(int width, int height) = 0;
    virtual ~HUDDrawable();
};

class ButtonDrawable : public HUDDrawable {
public:
    ButtonDrawable(
        const std::shared_ptr<AbstractFileReader> &file_reader,
        std::function<button(void)> get_input, glm::vec2 center_px, float size_px);

    void draw(int width, int height) override;

private:
    std::function<button(void)> get_input;

    std::shared_ptr<Program> program;

    float center_x, center_y;
    float size;

    int nb_points;
};

class JoyStickDrawable : public HUDDrawable {
public:
    JoyStickDrawable(
        const std::shared_ptr<AbstractFileReader> &file_reader,
        std::function<joystick(void)> get_input_px, glm::vec2 center_px, float size_px,
        float stick_size_px);

    void draw(int width, int height) override;

    ~JoyStickDrawable() override;

private:
    std::function<joystick(void)> get_input;

    std::shared_ptr<Program> program;

    float center_x, center_y;
    float size, stick_size;

    int nb_point_bound, nb_point_stick;
};

#endif// PHYVR_HUD_H
