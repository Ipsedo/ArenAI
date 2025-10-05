//
// Created by samuel on 06/09/18.
//

#ifndef PHYVR_COMPASS_H
#define PHYVR_COMPASS_H

#include "../entity/base.h"
#include "../graphics/drawable/triangle.h"
#include "../graphics/misc.h"

class Compass : public Drawable {
private:
  Base *target;
  Triangle triangle;

public:
  Compass(Base *target);

  void draw(draw_infos infos) override;
};

#endif// PHYVR_COMPASS_H
