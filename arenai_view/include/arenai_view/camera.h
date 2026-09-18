//
// Created by samuel on 18/03/2023.
//

#ifndef ARENAI_CAMERA_H
#define ARENAI_CAMERA_H

#include <functional>
#include <memory>
#include <numbers>
#include <optional>

#include <glm/glm.hpp>

namespace arenai::view {

    constexpr float DEFAULT_FOV = std::numbers::pi_v<float> / 4.f;

    class AbstractCamera {
    public:
        virtual ~AbstractCamera() = default;

        virtual glm::vec3 pos() = 0;
        virtual glm::vec3 look() = 0;
        virtual glm::vec3 up() = 0;

        virtual float fov();
        virtual glm::vec3 pivot();
    };

    class StaticCamera final : public AbstractCamera {
    public:
        StaticCamera(glm::vec3 pos, glm::vec3 look, glm::vec3 up);

        glm::vec3 pos() override;
        glm::vec3 look() override;
        glm::vec3 up() override;

    private:
        glm::vec3 pos_vec;
        glm::vec3 look_vec;
        glm::vec3 up_vec;
    };

    using RaycastFunction = std::function<std::optional<float>(glm::vec3 from, glm::vec3 to)>;

    class CollisionCamera final : public AbstractCamera {
    public:
        CollisionCamera(
            std::shared_ptr<AbstractCamera> inner, RaycastFunction ray_cast, float frame_period,
            float margin = 0.5f, float min_distance = 2.f, float extend_speed = 4.f);

        glm::vec3 pos() override;
        glm::vec3 look() override;
        glm::vec3 up() override;

        float fov() override;
        glm::vec3 pivot() override;

    private:
        std::shared_ptr<AbstractCamera> inner;
        RaycastFunction raycast;
        float frame_period;
        float margin;
        float min_distance;
        float extend_speed;
        float current_distance;
    };

}// namespace arenai::view

#endif// ARENAI_CAMERA_H
