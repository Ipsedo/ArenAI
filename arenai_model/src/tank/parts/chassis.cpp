//
// Created by samuel on 02/04/2023.
//

#include "./chassis.h"

#include <glm/gtc/type_ptr.hpp>

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    ChassisItem::ChassisItem(
        const std::string &prefix_name, JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const glm::vec3 position, const glm::vec3 scale, const float mass)
        : LifeItem(10), ConvexItem(
                            prefix_name + "_chassis", engine,
                            std::make_shared<ObjShape>(
                                file_reader, std::filesystem::path("obj") / "anubis_chassis.obj"),
                            // Bullet shifted the freshly created body two units
                            // down through setCenterOfMassTransform: same spawn
                            position + glm::vec3(0.f, -2.f, 0.f), scale, mass) {}

}// namespace arenai::model
