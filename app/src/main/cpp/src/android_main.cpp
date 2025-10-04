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

#include "./agent/executorch_agent.h"

// https://github.com/JustJokerX/NativeActivityFromJavaActivity/blob/master/app/src/main/cpp/main.cpp

static void on_cmd_wrapper(struct android_app *app, int32_t cmd) {
  auto *engine = (UserGameTanksEnvironment *)app->userData;
  engine->on_cmd(app, cmd);
}

static int32_t on_input_wrapper(struct android_app *app, AInputEvent *event) {
  auto *engine = (class UserGameTanksEnvironment *)app->userData;
  return engine->on_input(app, event);
}

typedef std::chrono::steady_clock steady_clock_t;
typedef std::chrono::duration<float> secs_f;

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

  const float target_fps = 60.0f;
  const secs_f frame_dt = secs_f(1.0f / target_fps);

  auto last_time = steady_clock_t::now();
  auto next_frame =
      last_time +
      std::chrono::duration_cast<steady_clock_t::duration>(frame_dt);

  while (true) {
    int ident;
    int events;
    struct android_poll_source *source;

    while ((ident = ALooper_pollOnce(env->is_running() ? 0 : -1, nullptr,
                                     &events, (void **)&source)) >= 0) {
      if (source != nullptr) {
        source->process(state, source);
      }
      if (state->destroyRequested != 0) {
        // delete env;
        LOG_INFO("closing PhyVR");
        return;
      }
    }

    auto now = steady_clock_t::now();
    auto elapsed_time = std::chrono::duration_cast<secs_f>(now - last_time);
    last_time = now;

    std::this_thread::sleep_for(frame_dt - elapsed_time);

    auto step_result = env->step(frame_dt.count(), {});
  }
}
