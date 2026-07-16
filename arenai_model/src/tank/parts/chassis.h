//
// Created by samuel on 02/04/2023.
//

#ifndef ARENAI_CHASSIS_H
#define ARENAI_CHASSIS_H

#include "../../items/convex.h"

namespace arenai::model {

    class ChassisItem final : public LifeItem, public ConvexItem {
    public:
        ChassisItem(
            const std::string &prefix_name,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            glm::vec3 position, glm::vec3 scale, float mass);
    };

}// namespace arenai::model

#endif// ARENAI_CHASSIS_H
