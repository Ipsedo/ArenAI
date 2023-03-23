//
// Created by samuel on 19/03/2023.
//

#include "./item.h"

#include <glm/gtc/type_ptr.hpp>

#include "../utils/asset.h"
#include "../utils/logging.h"

Item::Item(std::string name) : name(std::move(name)) {

}

std::string Item::get_name() {
    return name;
}

glm::mat4 Item::get_model_matrix() {
    btScalar tmp[16];
    btTransform tr;

    get_body()->getMotionState()->getWorldTransform(tr);

    tr.getOpenGLMatrix(tmp);

    return glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), _get_scale());
}
