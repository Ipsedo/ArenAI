//
// Created by samuel on 16/03/2023.
//

#include <android/sensor.h>

#include <android_native_app_glue.h>
#include <dlfcn.h>
#include <cassert>

#include "./utils/logging.h"
#include "./core.h"


// https://github.com/JustJokerX/NativeActivityFromJavaActivity/blob/master/app/src/main/cpp/main.cpp

/**
 * Process the next main command.
 */
static void engine_handle_cmd(struct android_app *app, int32_t cmd) {
    auto *engine = (CoreEngine *) app->userData;
    switch (cmd) {
        case APP_CMD_SAVE_STATE:
            // The system has asked us to save our current state.  Do so.
            /*app->savedState = malloc(sizeof(CoreEngine));
            // TODO real object copy...
            app->savedState = engine;
            app->savedStateSize = sizeof(CoreEngine);
            */
            LOG_INFO("save state");
            break;
        case APP_CMD_INIT_WINDOW:
            // The window is being shown, get it ready.
            if (app->window != nullptr) {
                LOG_INFO("opening window");
                engine->new_view(app->activity->assetManager, app->window);
            }
            break;
        case APP_CMD_TERM_WINDOW:
            engine->pause();
            LOG_INFO("close");
            break;
        case APP_CMD_GAINED_FOCUS:
            break;
        case APP_CMD_LOST_FOCUS:
            engine->pause();
            /*engine_draw_frame(engine);*/
            break;
        default:
            break;
    }
}

static int32_t on_input_wrapper(struct android_app *app, AInputEvent *event) {
    auto *engine = (class CoreEngine *) app->userData;
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
    engine = new CoreEngine(state->activity->assetManager, state->window);
    //}

    state->userData = engine;
    state->onAppCmd = engine_handle_cmd;
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
        while ((ident = ALooper_pollAll(
                engine->is_running() ? 0 : -1, nullptr,
                &events,
                (void **) &source)) >= 0) {

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