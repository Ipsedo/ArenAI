//
// Created by samuel on 21/03/2023.
//

#ifndef PHYVR_CORE_H
#define PHYVR_CORE_H

#include <random>
#include <android_native_app_glue.h>

#include "./model/engine.h"
#include "./view/renderer.h"
#include "./controller/controller.h"

class CoreEngine {
public:
    CoreEngine(AAssetManager *mgr);

    bool is_running() const;

    void draw();

    void step(float time_delta);

    int32_t on_input(struct android_app *app, AInputEvent *event);

    void on_cmd(struct android_app *app, int32_t cmd);

private:
    std::shared_ptr<StaticCamera> camera;

    PhysicEngine physic_engine;
    std::unique_ptr<Renderer> renderer;
    ControllerEngine controller_engine;

    std::vector<std::shared_ptr<Item>> items;

    std::random_device dev;
    std::mt19937 rng;

    bool is_paused;

    void _new_view(AAssetManager *mgr, ANativeWindow *window);

    void _pause();
};

#endif //PHYVR_CORE_H
