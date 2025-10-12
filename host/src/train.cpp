//
// Created by samuel on 03/10/2025.
//

#include "./train.h"

#include "./train_environment.h"
#include "./train_gl_context.h"
#include "./utils/linux_file_reader.h"
#include "./utils/replay_buffer.h"
#include "./utils/saver.h"
#include "phyvr_view/errors.h"

void train(
    const std::filesystem::path &output_folder, const std::filesystem::path &android_assets_path) {

    const auto env = std::make_unique<TrainTankEnvironment>(4, android_assets_path);

    env->reset_physics();
    env->reset_drawables(std::make_shared<TrainGlContext>());

    check_gl_error("reset");

    for (int i = 0; i < 100; i++) {
        auto promise = std::promise<std::vector<Action>>();
        promise.set_value(std::vector<Action>(4));
        auto future = promise.get_future();

        auto state_1 = env->step(1.f / 30.f, future);

        int j = 0;
        for (auto [s, _, __]: state_1)
            save_png_rgb(
                s.vision,
                output_folder / ("img_" + std::to_string(i) + "_" + std::to_string(j++) + ".png"));
    }
}
