//
// Created by samuel on 21/03/2023.
//

#include "./core.h"
#include "./view/drawable/specular.h"
#include "./view/drawable/cubemap.h"
#include "utils/logging.h"

#include <glm/gtx/transform.hpp>

CoreEngine::CoreEngine(AAssetManager *mgr, ANativeWindow *window) :
        is_paused(true),
        camera(std::make_shared<StaticCamera>(
                glm::vec3(0., 10., 0.),
                glm::vec3(0., 10., 1.),
                glm::vec3(0., 1., 0.))
        ),
        physic_engine(),
        controller_engine(),
        items(),
        dev(),
        rng(dev()) {

    auto map = std::make_shared<HeightMapItem>(
            "height_map",
            mgr,
            "heightmap/heightmap6_2.png",
            glm::vec3(0., 0., 0.),
            glm::vec3(4., 40., 4.)
    );

    items.push_back(map);
    physic_engine.add_item(map);
}

void CoreEngine::new_view(AAssetManager *mgr, ANativeWindow *window) {
    renderer = std::make_unique<Renderer>(window, camera);

    std::uniform_real_distribution<float> u_dist(0., 1.);

    renderer->add_drawable("cubemap", std::make_shared<CubeMap>(
            mgr,
            "cubemap/1"
    ));

    for (const auto &item: items) {
        glm::vec4 color(
                u_dist(rng),
                u_dist(rng),
                u_dist(rng),
                1.f
        );

        auto drawable = std::make_shared<Specular>(
                mgr,
                item->get_shape()->get_vertices(),
                item->get_shape()->get_normals(),
                color,
                color,
                color,
                50.f
        );

        renderer->add_drawable(item->get_name(), drawable);
    }

    is_paused = false;
}

void CoreEngine::draw() {
    if (is_paused)
        return;

    std::vector<std::tuple<std::string, glm::mat4>> model_matrices{};
    std::transform(
            items.begin(), items.end(),
            std::back_inserter(model_matrices),
            [](const std::shared_ptr<Item> &item) {
                return std::tuple<std::string, glm::mat4>(item->get_name(),
                                                          item->get_model_matrix());
            });

    model_matrices.emplace_back(
            "cubemap",
            glm::scale(glm::mat4(1.), glm::vec3(2000., 2000., 2000.))
    );

    renderer->draw(model_matrices);
}

void CoreEngine::step(float time_delta) {
    if (is_paused) return;

    physic_engine.step(time_delta);
}

int32_t CoreEngine::on_input(struct android_app *app, AInputEvent *event) {
    return controller_engine.on_event(event);
}

void CoreEngine::pause() {
    is_paused = true;
    renderer = std::nullptr_t();
}

bool CoreEngine::is_running() const {
    return !is_paused;
}
