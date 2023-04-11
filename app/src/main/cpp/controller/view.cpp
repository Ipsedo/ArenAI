//
// Created by samuel on 28/03/2023.
//

#include "./view.h"

#include "../utils/units.h"

/*
 * View
 */

View::View(int screen_width, int screen_height)
    : screen_width(screen_width), screen_height(screen_height), dimens_dps() {}

void View::add_dimen(const std::string &dimen_name, int dimen_dp) {
  dimens_dps.insert({dimen_name, dimen_dp});
}

float View::get_pixel(AConfiguration *config, const std::string &dimen_name) {
  return dp_to_px(config, dimens_dps[dimen_name]);
}

/*
 * LinearLayout
 */

LinearLayout::LinearLayout(int screen_width, int screen_height,
                           LinearLayout::ORIENTATION orientation)
    : View(screen_width, screen_height), orientation(orientation) {}

void LinearLayout::add_view(const std::shared_ptr<View> &view) {
  views.push_back(view);
}

float LinearLayout::get_width() { return 0; }

float LinearLayout::get_height() { return 0; }

float LinearLayout::get_margin() { return 0; }

void LinearLayout::set_width(float width) {}

void LinearLayout::set_height(float height) {}

void LinearLayout::build() {}

/*
 * CornerLayout
 */

CornerLayout::CornerLayout(int screen_width, int screen_height)
    : View(screen_width, screen_height) {}

void CornerLayout::add_view(const std::shared_ptr<View> &view,
                            CornerLayout::CORNER corner) {
  views.insert({corner, view});
}

float CornerLayout::get_width() { return 0; }

float CornerLayout::get_height() { return 0; }

float CornerLayout::get_margin() { return 0; }

void CornerLayout::set_width(float width) {}

void CornerLayout::set_height(float height) {}

void CornerLayout::build() {}
