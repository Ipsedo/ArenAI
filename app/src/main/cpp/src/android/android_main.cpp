//
// Created by samuel on 16/03/2023.
//

#include <android/sensor.h>

#include <android_native_app_glue.h>
#include <cassert>
#include <chrono>
#include <dlfcn.h>
#include <thread>

#include "./game_environment.h"
#include <phyvr_utils/logging.h>

// https://github.com/JustJokerX/NativeActivityFromJavaActivity/blob/master/app/src/main/cpp/main.cpp

static void on_cmd_wrapper(struct android_app *app, int32_t cmd) {
  auto *engine = (UserGameTanksEnvironment *)app->userData;
  engine->on_cmd(app, cmd);
}

static int32_t on_input_wrapper(struct android_app *app, AInputEvent *event) {
  auto *engine = (class UserGameTanksEnvironment *)app->userData;
  return engine->on_input(app, event);
}

void android_main(struct android_app *state) {
  UserGameTanksEnvironment *env;

  /*if (state->savedState != nullptr) {
      // We are starting with a previous saved state; restore from it.
      env = (UserGameTanksEnvironment *) state->savedState;
      LOG_INFO("load state");
  } else {*/
  env = new UserGameTanksEnvironment(state);
  //}

  env->reset_physics();

  state->userData = env;
  state->onAppCmd = on_cmd_wrapper;
  state->onInputEvent = on_input_wrapper;

  std::clock_t last_time = std::clock();

  while (true) {
    int ident;
    int events;
    struct android_poll_source *source;

    while ((ident = ALooper_pollOnce(env->is_running() ? 0 : -1, nullptr,
                                     &events, (void **)&source)) >= 0) {

      if (source != nullptr) {
        source->process(state, source);
      }

      // If a sensor has data, process it now.
      if (ident == LOOPER_ID_USER) {
      }

      if (state->destroyRequested != 0) {
        // delete env;
        LOG_INFO("closing PhyVR");
        return;
      }
    }

    env->step(1.f / 30.f, {});

    std::clock_t now = std::clock();
    auto delta = std::chrono::milliseconds(
        1000L / 30L - 1000L * (now - last_time) / CLOCKS_PER_SEC);
    delta = std::max(delta, std::chrono::milliseconds(0L));

    std::this_thread::sleep_for(delta);

    last_time = now;
  }
}
