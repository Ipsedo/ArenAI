//
// Created by samuel on 28/05/18.
//

#ifndef PHYVR_CONTROLS_H
#define PHYVR_CONTROLS_H

struct input {
  float xAxis;
  float speed;
  bool brake;
  float turretDir;
  float turretUp;
  bool respawn;
  bool fire;
};

struct output {
  bool vibrate;
};

class Controls {
public:
  virtual void onInput(input in) = 0;
  virtual output getOutput() = 0;
};

#endif// PHYVR_CONTROLS_H
