//
// Created by samuel on 28/05/18.
//

#ifndef PHYVR_CAR_H
#define PHYVR_CAR_H


#include "../base.h"
#include <glm/glm.hpp>

class Car : public Base {
public:
    Car();
    void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;
    ~Car();
private:
    void init();
};


#endif //PHYVR_CAR_H
