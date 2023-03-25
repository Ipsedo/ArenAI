//
// Created by samuel on 16/03/2023.
//

#include <android/sensor.h>

#include <android_native_app_glue.h>
#include <cassert>
#include <dlfcn.h>

#include "./core.h"
#include "./utils/logging.h"

// https://github.com/JustJokerX/NativeActivityFromJavaActivity/blob/master/app/src/main/cpp/main.cpp

/**
 * Process the next main command.
 */
static void on_cmd_wrapper(struct android_app *app, int32_t cmd) {
  auto *engine = (CoreEngine *)app->userData;
  engine->on_cmd(app, cmd);
}

static int32_t on_input_wrapper(struct android_app *app, AInputEvent *event) {
  auto *engine = (class CoreEngine *)app->userData;
  return engine->on_input(app, event);
}

/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(struct android_app *state) {
  CoreEngine *engine;

  /*if (state->savedState != nullptr) {
      // We are starting with a previous saved state; restore from it.
      engine = (CoreEngine *) state->savedState;
      LOG_INFO("load state");
  } else {*/
  engine = new CoreEngine(state->activity->assetManager);
  //}

  state->userData = engine;
  state->onAppCmd = on_cmd_wrapper;
  state->onInputEvent = on_input_wrapper;

  // loop waiting for stuff to do.

  while (true) {
    // Read all pending events.
    int ident;
    int events;
    struct android_poll_source *source;

    // If not animating, we will block forever waiting for events.
    // If animating, we loop until all events are read, then continue
    // to draw the next frame of animation.
    while ((ident = ALooper_pollAll(engine->is_running() ? 0 : -1, nullptr,
                                    &events, (void **)&source)) >= 0) {

      // Process this event.
      if (source != nullptr) {
        source->process(state, source);
      }

      // If a sensor has data, process it now.
      if (ident == LOOPER_ID_USER) {
      }

      // Check if we are exiting.
      if (state->destroyRequested != 0) {
        delete engine;
        LOG_INFO("closing PhyVR");
        return;
      }
    }

    engine->step(1.f / 60.f);
    engine->draw();
  }
}