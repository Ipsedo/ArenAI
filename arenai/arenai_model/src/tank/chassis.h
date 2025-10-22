//
// Created by samuel on 02/04/2023.
//

#ifndef ARENAI_CHASSIS_H
#define ARENAI_CHASSIS_H

#include <arenai_model/convex.h>

class ChassisItem final : public LifeItem, public ConvexItem {
public:
    ChassisItem(
        const std::string &prefix_name, const std::shared_ptr<AbstractFileReader> &file_reader,
        glm::vec3 position, glm::vec3 scale, float mass);
};

#endif// ARENAI_CHASSIS_H
