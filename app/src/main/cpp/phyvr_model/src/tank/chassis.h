//
// Created by samuel on 02/04/2023.
//

#ifndef PHYVR_CHASSIS_H
#define PHYVR_CHASSIS_H

#include <phyvr_model/convex.h>

class ChassisItem : public ConvexItem {
public:
  ChassisItem(const std::string &prefix_name,
              const std::shared_ptr<AbstractFileReader> &file_reader,
              glm::vec3 position, glm::vec3 scale, float mass);
};

#endif // PHYVR_CHASSIS_H
