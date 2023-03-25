//
// Created by samuel on 23/03/20.
//

#include "solid_curve.h"
#include "../graphics/misc.h"
#include "../utils/gl_utils.h"
#include "../utils/shader.h"
#include "glm/gtc/type_ptr.hpp"

std::string vertex_shader = "uniform mat4 uVPMatrix;\n"
                            "attribute vec4 vPosition;\n"
                            "\n"
                            "void main() {\n"
                            "    gl_Position = uVPMatrix * vPosition;\n"
                            "}";

std::string fragment_shader = "precision mediump float;\n"
                              "\n"
                              "uniform vec4 vColor;\n"
                              "\n"
                              "void main() {\n"
                              "    gl_FragColor = vColor;\n"
                              "}";

void Curve::draw(draw_infos infos) {
  btRigidBody *body = canon->getMissileCopy();

  world->addRigidBody(body);

  std::vector<float> pts;

  for (int i = 0; i < nb_pts; i++) {

    btVector3 pos = body->getWorldTransform().getOrigin();
    pts.push_back(pos.x());
    pts.push_back(pos.y());
    pts.push_back(pos.z());
    pts.push_back(1.f);

    world->stepSimulation(1.f / 60.f);
  }

  world->removeRigidBody(body);
  delete body->getMotionState();
  delete body;

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glUseProgram(mProgram);

  glLineWidth(3);

  glEnableVertexAttribArray(mPositionHandle);
  glVertexAttribPointer(mPositionHandle, 4, GL_FLOAT, GL_FALSE, 4 * 4,
                        pts.data());

  float color[4]{1.f, 0.f, 0.f, 0.5f};
  glUniform4fv(mColorHandle, 1, color);

  glm::mat4 vp_matrix = infos.proj_matrix * infos.view_matrix;

  glUniformMatrix4fv(mVPMatrixHandle, 1, GL_FALSE, glm::value_ptr(vp_matrix));

  glDrawArrays(GL_LINES, 0, pts.size() / 4);

  glDisable(GL_BLEND);
}

Curve::Curve(Canon *canon) : canon(canon) {
  mProgram = glCreateProgram();
  vertexShader = loadShader(GL_VERTEX_SHADER, vertex_shader.c_str());
  fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragment_shader.c_str());
  glAttachShader(mProgram, vertexShader);
  glAttachShader(mProgram, fragmentShader);
  glLinkProgram(mProgram);

  mPositionHandle = (GLuint)glGetAttribLocation(mProgram, "vPosition");
  mColorHandle = (GLuint)glGetUniformLocation(mProgram, "vColor");
  mVPMatrixHandle = (GLuint)glGetUniformLocation(mProgram, "uVPMatrix");

  collisionConfiguration = new btDefaultCollisionConfiguration();
  dispatcher = new btCollisionDispatcher(collisionConfiguration);
  broadPhase = new btDbvtBroadphase();
  constraintSolver = new btSequentialImpulseConstraintSolver();

  world = new btDiscreteDynamicsWorld(dispatcher, broadPhase, constraintSolver,
                                      collisionConfiguration);
  world->setGravity(btVector3(0, -10.f, 0));
}
