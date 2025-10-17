//
// Created by samuel on 26/03/2023.
//

#ifndef ARENAI_EVENT_H
#define ARENAI_EVENT_H

#include <android/input.h>

class EventHandler {
public:
    virtual bool on_event(AInputEvent *event) = 0;
};

class PointerLocker {
public:
    virtual int get_pointer_id() = 0;
};

#endif// ARENAI_EVENT_H
