//
// Created by samuel on 21/03/2023.
//

#ifndef PHYVR_CORE_H
#define PHYVR_CORE_H

#include <random>

#include "./model/engine.h"
#include "./view/renderer.h"
#include "./controller/controller.h"

class CoreEngine {
public:
    CoreEngine(AAssetManager *mgr, ANativeWindow *window);

    void new_view(AAssetManager *mgr, ANativeWindow *window);
    void pause();

    bool is_running() const;

    void draw();
    void step(float time_delta);
    int32_t on_input(struct android_app *app, AInputEvent *event);
private:
    std::shared_ptr<StaticCamera> camera;

    PhysicEngine physic_engine;
    std::unique_ptr<Renderer> renderer;
    ControllerEngine controller_engine;

    std::vector<std::shared_ptr<Item>> items;

    std::random_device dev;
    std::mt19937 rng;

    bool is_paused;
};

#endif //PHYVR_CORE_H
