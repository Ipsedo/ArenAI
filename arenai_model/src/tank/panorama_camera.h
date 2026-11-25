#ifndef ARENAI_PANORAMA_CAMERA_H
#define ARENAI_PANORAMA_CAMERA_H

#include <memory>
#include <vector>

#include <arenai_model/item.h>
#include <arenai_view/camera.h>

namespace arenai::model {

    // level camera above the canon, turned by a fixed yaw from the canon heading
    class PanoramaCamera final : public view::AbstractCamera {
    public:
        PanoramaCamera(std::shared_ptr<Item> canon, float yaw_offset);

        glm::vec3 pos() override;
        glm::vec3 look() override;
        glm::vec3 up() override;

        float fov() override;
        glm::vec3 pivot() override;

    private:
        std::shared_ptr<Item> canon;
        // rad
        float yaw_offset;
    };

    // PANORAMA_VIEWS cameras covering the full turn, left to right, the canon heading in the middle
    std::vector<std::shared_ptr<view::AbstractCamera>>
    make_panorama_cameras(const std::shared_ptr<Item> &canon);

}// namespace arenai::model

#endif//ARENAI_PANORAMA_CAMERA_H
