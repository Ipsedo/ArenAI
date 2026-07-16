//
// Created by samuel on 18/03/2023.
//

#ifndef ARENAI_CAMERA_H
#define ARENAI_CAMERA_H

#include <functional>
#include <memory>
#include <optional>

#include <glm/glm.hpp>

namespace arenai::view {

    class AbstractCamera {
    public:
        virtual ~AbstractCamera() = default;

        virtual glm::vec3 pos() = 0;

        virtual glm::vec3 look() = 0;

        virtual glm::vec3 up() = 0;
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

    // Fraction of the [from -> to] segment at which the world is first hit, in
    // (0, 1]. std::nullopt when the path is free.
    using RaycastFunction = std::function<std::optional<float>(glm::vec3 from, glm::vec3 to)>;

    /**
     * Spring-arm decorator: keeps the wrapped camera's aim but pulls its position
     * toward the look-at pivot when world geometry blocks the [pivot -> pos]
     * segment, so the camera never goes behind walls or under the terrain.
     * Retraction is instantaneous (no clipping), extension is smoothed.
     */
    class CollisionCamera final : public AbstractCamera {
    public:
        CollisionCamera(
            std::shared_ptr<AbstractCamera> inner, RaycastFunction raycast, float frame_period,
            float margin = 0.5f, float min_distance = 2.f, float extend_speed = 4.f);

        glm::vec3 pos() override;

        glm::vec3 look() override;

        glm::vec3 up() override;

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
