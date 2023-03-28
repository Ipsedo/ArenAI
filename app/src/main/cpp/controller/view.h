//
// Created by samuel on 28/03/2023.
//

#ifndef PHYVR_VIEW_H
#define PHYVR_VIEW_H

#include <android/configuration.h>
#include <map>
#include <string>
#include <vector>

class View {
public:
  View(int screen_width, int screen_height);
  void add_dimen(const std::string &dimen_name, int dimen_dp);
  float get_pixel(AConfiguration *config, const std::string &dimen_name);
  virtual float get_width() = 0;
  virtual float get_height() = 0;
  virtual float get_margin() = 0;

private:
  std::map<std::string, int> dimens_dps;

protected:
  const int screen_width, screen_height;

  virtual void set_width(float width) = 0;
  virtual void set_height(float height) = 0;
};

class LinearLayout : public View {
public:
  enum START_FROM {
    LEFT_TOP = 0,
    RIGHT_TOP = 1,
    RIGHT_BOTTOM = 2,
    LEFT_BOTTOM = 3
  };

  enum ORIENTATION { HORIZONTAL = 0, VERTICAL = 1 };

  LinearLayout(int screen_width, int screen_height,
               LinearLayout::START_FROM start_from,
               LinearLayout::ORIENTATION orientation);
  void add_view(std::shared_ptr<View> view);
  void build();

private:
  std::vector<std::shared_ptr<View>> views;
  LinearLayout::START_FROM start_from;
  LinearLayout::ORIENTATION orientation;
};

#endif // PHYVR_VIEW_H
