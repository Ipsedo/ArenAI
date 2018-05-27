//
// Created by samuel on 27/05/18.
//

#include "convex.h"

#include <glm/gtc/quaternion.hpp>
#include <android/log.h>
#include <glm/gtc/matrix_transform.hpp>

#include "../utils/string_utils.h"
#include "../utils/assets.h"

Convex::Convex(AAssetManager *mgr, std::string objFileName, glm::vec3 pos, glm::vec3 scale,
               glm::mat4 rotationMatrix, float mass) {

    std::string objTxt = getFileText(mgr, objFileName);

    modelVBO = new ModelVBO(
            objTxt,
            new float[4]{ (float) rand() / RAND_MAX,
                          (float) rand() / RAND_MAX,
                          (float) rand() / RAND_MAX,
                          1.f});

    collisionShape = parseObj(objTxt);
    //collisionShape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));

    myTransform.setIdentity();
    myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
    glm::quat tmp = glm::quat_cast(rotationMatrix);
    myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

    btVector3 intertie(0.f, 0.f, 0.f);
    if (mass)
        collisionShape->calculateLocalInertia(mass, intertie);

    defaultMotionState = new btDefaultMotionState(myTransform);

    btRigidBody::btRigidBodyConstructionInfo constrInfo(0.f,
                                                        defaultMotionState,
                                                        collisionShape,
                                                        intertie);

    rigidBody = new btRigidBody(constrInfo);
}

btBvhTriangleMeshShape* Convex::parseObj(std::string objFileText) {
    vector<std::string> lines = split(objFileText, '\n');
    __android_log_print(ANDROID_LOG_VERBOSE, "POIR", "a %ld", lines.size());

    vector<float> vertex_list;
    vector<int> vertex_draw_order;

    for (auto str : lines) {
        vector<std::string> splitted_line = split(str, ' ');
        if(!splitted_line.empty()) {
            if (splitted_line[0] == "v") {
                vertex_list.push_back(std::stof(splitted_line[1]));
                vertex_list.push_back(std::stof(splitted_line[2]));
                vertex_list.push_back(std::stof(splitted_line[3]));
            } else if (splitted_line[0] == "f") {
                vector<string> v1 = split(splitted_line[1], '/');
                vector<string> v2 = split(splitted_line[2], '/');
                vector<string> v3 = split(splitted_line[3], '/');

                vertex_draw_order.push_back(std::stoi(v1[0]) - 1);
                vertex_draw_order.push_back(std::stoi(v2[0]) - 1);
                vertex_draw_order.push_back(std::stoi(v3[0]) - 1);

                v1.clear();
                v2.clear();
                v3.clear();
            }
        }
        splitted_line.clear();
    }




    unsigned long nbVertex = vertex_draw_order.size();
    /*btScalar *mesh = new btScalar[nbVertex * 3];
    for (int i = 0; i < nbVertex; i++) {
        btVector3 point(vertex_list[(vertex_draw_order[i] - 1) * 3],
                        vertex_list[(vertex_draw_order[i] - 1) * 3 + 1],
                        vertex_list[(vertex_draw_order[i] - 1) * 3 + 2]
        );
        mesh[i * 3] = vertex_list[(vertex_draw_order[i] - 1) * 3];
        mesh[i * 3 + 1] = vertex_list[(vertex_draw_order[i] - 1) * 3 + 1];
        mesh[i * 3 + 2] = vertex_list[(vertex_draw_order[i] - 1) * 3 + 2];
    }*/


    btScalar *pts = new btScalar[vertex_list.size()];
    for (int i = 0; i < vertex_list.size(); i++) {
        pts[i] = vertex_list[i];
        __android_log_print(ANDROID_LOG_VERBOSE, "POIR", "a %f", pts[i]);
    }

    btTriangleIndexVertexArray*  indexVertexArrays =
            new btTriangleIndexVertexArray (int(nbVertex) / 3, &vertex_draw_order[0], 3 * sizeof(int), int(nbVertex), &pts[0], 3 * sizeof(btScalar));
    btBvhTriangleMeshShape *shape = new btBvhTriangleMeshShape(indexVertexArrays, true);

    //btConvexHullShape* shape = new btConvexHullShape(mesh, int(nbVertex), sizeof(btScalar) * 3);
    //__android_log_print(ANDROID_LOG_VERBOSE, "POIR", "b %d %ld %d %d", shape->getNumPlanes(), nbVertex, shape->getNumPoints(), shape->getNumVertices());
    return shape;
}

void Convex::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lightPos) {
    /*btScalar tmp[16];
    defaultMotionState->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
    for (int i = 0; i < 16; i++) {
        __android_log_print(ANDROID_LOG_VERBOSE, "POIR", "f %f", tmp[i]);
    }*/

    std::tuple<glm::mat4, glm::mat4> matrixes = getMatrixes(pMatrix, vMatrix);
    modelVBO->draw(std::get<0>(matrixes), std::get<1>(matrixes), lightPos);
}
