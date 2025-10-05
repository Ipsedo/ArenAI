//
// Created by samuel on 23/08/18.
//

#ifndef PHYVR_CUBEMAP_H
#define PHYVR_CUBEMAP_H

#include <string>

#include <android/asset_manager.h>
#include <GLES3/gl3.h>

#include "../misc.h"

using namespace std;

class CubeMap : public Drawable {
private:
  GLuint mProgram;
  GLuint vertexShader;
  GLuint fragmentShader;

  GLuint texCoordHandle;
  GLuint mvpMatrixHandle;
  GLuint samplerCubeHandle;

  GLuint textures;

  float vertices[6 * 2 * 3 * 3] = {-1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  -1.f, -1.f,
                                   1.f,  -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f, 1.f,  -1.f,

                                   -1.f, -1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  -1.f,
                                   -1.f, 1.f,  -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f, 1.f,

                                   1.f,  -1.f, -1.f, 1.f,  -1.f, 1.f,  1.f,  1.f,  1.f,
                                   1.f,  1.f,  1.f,  1.f,  1.f,  -1.f, 1.f,  -1.f, -1.f,

                                   -1.f, -1.f, 1.f,  -1.f, 1.f,  1.f,  1.f,  1.f,  1.f,
                                   1.f,  1.f,  1.f,  1.f,  -1.f, 1.f,  -1.f, -1.f, 1.f,

                                   -1.f, 1.f,  -1.f, 1.f,  1.f,  -1.f, 1.f,  1.f,  1.f,
                                   1.f,  1.f,  1.f,  -1.f, 1.f,  1.f,  -1.f, 1.f,  -1.f,

                                   -1.f, -1.f, -1.f, -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f,
                                   1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  1.f,  -1.f, 1.f};

  int nbTriangle = 6 * 2 * 3;

  glm::mat4 modelMatrix;

  void bind();

  void makeProgram();

  void loadCubeMaptexture(AAssetManager *mgr, string cubaMapAssetPath);

public:
  CubeMap(AAssetManager *mgr, string cubeMapAssetPath, float sideLength);

  void draw(draw_infos infos) override;

  ~CubeMap();
};

#endif// PHYVR_CUBEMAP_H
