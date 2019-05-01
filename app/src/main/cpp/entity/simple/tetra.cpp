//
// Created by samuel on 21/04/19.
//

#include "tetra.h"

Tetra::Tetra(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass)
        : Poly([mgr](glm::vec3 scale) {
    std::string obj_str = getFileText(mgr, "obj/tetra.obj");
    return parseObj(move(obj_str));
}, Poly::makeModel(mgr, "obj/tetra.obj"), pos, scale, rotMat, mass, true) {

}

Tetra::Tetra(DiffuseModel *modelVBO, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass)
        : Poly([](glm::vec3 scale) {
    return parseObj("v -0.288675 -0.204528 -0.500000\n"
                    "v 0.577350 -0.204528 -0.000000\n"
                    "v -0.288675 -0.204528 0.500000\n"
                    "v 0.000000 0.613584 0.000000\n"
                    "vn -0.0000 -1.0000 0.0000\n"
                    "vn -0.9430 0.3327 0.0000\n"
                    "vn 0.4715 0.3327 0.8167\n"
                    "vn 0.4715 0.3327 -0.8167\n"
                    "usemtl Default_OBJ.001\n"
                    "s off\n"
                    "f 1//1 2//1 3//1\n"
                    "f 3//2 4//2 1//2\n"
                    "f 2//3 4//3 3//3\n"
                    "f 1//4 4//4 2//4");
}, modelVBO, pos, scale, rotMat, mass, false) {}
