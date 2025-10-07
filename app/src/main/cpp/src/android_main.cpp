//
// Created by samuel on 16/03/2023.
//

#include <cassert>
#include <chrono>
#include <future>
#include <thread>

#include <android/sensor.h>
#include <android_native_app_glue.h>

#include <phyvr_utils/logging.h>

#include "./agent/executorch_agent.h"
#include "./game_environment.h"

// https://github.com/JustJokerX/NativeActivityFromJavaActivity/blob/master/app/src/main/cpp/main.cpp

static void on_cmd_wrapper(struct android_app *app, int32_t cmd) {
    auto *engine = (UserGameTanksEnvironment *) app->userData;
    engine->on_cmd(app, cmd);
}

static int32_t on_input_wrapper(struct android_app *app, AInputEvent *event) {
    auto *engine = (class UserGameTanksEnvironment *) app->userData;
    return engine->on_input(app, event);
}

typedef std::chrono::steady_clock steady_clock_t;
typedef std::chrono::duration<float> secs_f;

void android_main(struct android_app *app) {
    const float target_fps = 30.0f;
    const secs_f frame_dt = secs_f(1.0f / target_fps);

    constexpr int nb_tanks = 2;
    bool will_quit = false;

    auto env = std::make_unique<UserGameTanksEnvironment>(app, nb_tanks, frame_dt.count());
    auto agent = std::make_unique<ExecuTorchAgent>(app, "executorch/actor.pte");

    app->userData = env.get();
    app->onAppCmd = on_cmd_wrapper;
    app->onInputEvent = on_input_wrapper;

    auto last_time = steady_clock_t::now();

    auto agents_state = env->reset_physics();

    auto action_promise = std::promise<std::vector<Action>>();
    action_promise.set_value(agent->act(agents_state));
    auto action_future = action_promise.get_future();

    while (!will_quit) {
        int ident;
        int events;
        struct android_poll_source *source;

        while ((ident = ALooper_pollOnce(
                    env->is_running() ? 0 : -1, nullptr, &events, (void **) &source))
               >= 0) {
            if (source != nullptr) source->process(app, source);

            if (app->destroyRequested != 0) {
                will_quit = true;
                break;
            }
        }

        auto now = steady_clock_t::now();
        auto elapsed_time = std::chrono::duration_cast<secs_f>(now - last_time);
        last_time = now;

        std::this_thread::sleep_for(frame_dt - elapsed_time);

        auto step_result = env->step(frame_dt.count(), action_future);

        agents_state.clear();
        std::transform(
            step_result.begin(), step_result.end(), std::back_inserter(agents_state),
            [](auto t) { return std::get<0>(t); });

        action_future = std::async([agents_state, &agent]() { return agent->act(agents_state); });
    }

    app->userData = nullptr;
    app->onAppCmd = nullptr;
    app->onInputEvent = nullptr;

    UserGameTanksEnvironment::reset_singleton();
    env.reset();
    agent.reset();

    eglTerminate(eglGetDisplay(EGL_DEFAULT_DISPLAY));
}
