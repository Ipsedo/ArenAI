//
// Created by samuel on 26/03/2023.
//

#ifndef PHYVR_EVENT_H
#define PHYVR_EVENT_H

#include <android/input.h>

class EventHandler {
public:
  virtual bool on_event(AInputEvent *event) = 0;
};

class PointerLocker {
public:
  virtual int get_pointer_id() = 0;
};

#endif// PHYVR_EVENT_H
