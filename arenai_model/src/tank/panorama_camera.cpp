#include "./panorama_camera.h"

#include <cmath>
#include <numbers>

#include <arenai_model/constants.h>

namespace arenai::model {

    PanoramaCamera::PanoramaCamera(std::shared_ptr<Item> canon, const float yaw_offset)
        : canon(std::move(canon)), yaw_offset(yaw_offset) {}

    glm::vec3 PanoramaCamera::pos() {
        return glm::vec3(canon->get_model_matrix()[3])
               + glm::vec3(0.f, PANORAMA_CAMERA_HEIGHT, 0.f);
    }

    glm::vec3 PanoramaCamera::look() {
        const glm::vec3 axis = glm::mat3(canon->get_model_matrix()) * glm::vec3(0.f, 0.f, 1.f);
        const float yaw = std::atan2(axis.x, axis.z) + yaw_offset;

        return pos() + glm::vec3(std::sin(yaw), 0.f, std::cos(yaw));
    }

    glm::vec3 PanoramaCamera::up() { return {0.f, 1.f, 0.f}; }

    // vertical field of a view spanning one turn / PANORAMA_VIEWS horizontally
    float PanoramaCamera::fov() {
        const float half_horizontal =
            std::numbers::pi_v<float> / static_cast<float>(PANORAMA_VIEWS);

        return 2.f * std::atan(std::tan(half_horizontal) / PANORAMA_VIEW_ASPECT);
    }

    glm::vec3 PanoramaCamera::pivot() { return pos(); }

    std::vector<std::shared_ptr<view::AbstractCamera>>
    make_panorama_cameras(const std::shared_ptr<Item> &canon) {
        std::vector<std::shared_ptr<view::AbstractCamera>> cameras;
        cameras.reserve(PANORAMA_VIEWS);

        for (int view = 0; view < PANORAMA_VIEWS; view++)
            cameras.push_back(std::make_shared<PanoramaCamera>(
                canon, (static_cast<float>(PANORAMA_VIEWS - 1) / 2.f - static_cast<float>(view))
                           * 2.f * std::numbers::pi_v<float> / static_cast<float>(PANORAMA_VIEWS)));

        return cameras;
    }

}// namespace arenai::model
