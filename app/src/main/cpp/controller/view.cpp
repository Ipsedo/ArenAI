//
// Created by samuel on 28/03/2023.
//

#include "./view.h"

#include "../utils/units.h"

View::View(int screen_width, int screen_height)
    : screen_width(screen_width), screen_height(screen_height), dimens_dps() {}

void View::add_dimen(const std::string &dimen_name, int dimen_dp) {
  dimens_dps.insert({dimen_name, dimen_dp});
}

float View::get_pixel(AConfiguration *config, const std::string &dimen_name) {
  return dp_to_px(config, dimens_dps[dimen_name]);
}

LinearLayout::LinearLayout(int screen_width, int screen_height,
                           LinearLayout::START_FROM start_from,
                           LinearLayout::ORIENTATION orientation)
    : View(screen_width, screen_height), start_from(start_from),
      orientation(orientation) {}

void LinearLayout::add_view(std::shared_ptr<View> view) {
  views.push_back(view);
}

void LinearLayout::build() {

  int x_factor, y_factor;
  switch (orientation) {
  case VERTICAL:
    x_factor = 0, y_factor = 1;
  case HORIZONTAL:
    x_factor = 1, y_factor = 0;
  }

  float start_x, start_y;

  switch (start_from) {
  case LEFT_TOP:
    start_x = 0.f, start_y = 0.f;
  case LEFT_BOTTOM:
    start_x = 0.f, start_y = float(screen_height);
    y_factor *= -1;
  case RIGHT_TOP:
    start_x = float(screen_width), start_y = 0.f;
    x_factor *= -1;
  case RIGHT_BOTTOM:
    start_x = float(screen_width), start_y = float(screen_height);
    x_factor *= -1;
    y_factor *= -1;
  }

  if ((orientation == VERTICAL && y_factor < 0) ||
      (orientation == HORIZONTAL && x_factor < 0))
    std::reverse(views.begin(), views.end());

  float curr_px = orientation == VERTICAL ? start_y : start_x;

  for (auto &view : views) {
    float view_width = view->get_width();
    float view_height = view->get_height();
    float view_margin = view->get_margin();

    if (orientation == VERTICAL) {

    } else {
      // horizontal
    }
  }
}
