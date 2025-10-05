//
// Created by samuel on 29/10/18.
//

#include "normalmodel.h"

#include "../../utils/assets.h"
#include "../../utils/gl_utils.h"
#include "../../utils/image.h"
#include "../../utils/shader.h"
#include "../../utils/string_utils.h"
#include "glm/gtc/type_ptr.hpp"

NormalMapModel::NormalMapModel(
  AAssetManager *mgr, string objFile, string textureFile, string normalsFile)
    : textures(new GLuint[2]{0, 0}), nbVertex(0) {
  initPrgm(mgr);
  bind();
  initTex(mgr, move(textureFile), move(normalsFile));
  genBuffer(mgr, move(objFile));
}

void NormalMapModel::initTex(AAssetManager *mgr, string textureFile, string normalsFile) {
  glGenTextures(2, textures);

  libpng_image tmp1 = readPNG(mgr, textureFile);
  imgRGB img1 = toImgRGB(tmp1);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures[0]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(
    GL_TEXTURE_2D, 0, GL_RGB, img1.width, img1.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img1.pixels);

  delete[] tmp1.data;
  delete[] tmp1.rowPtrs;
  delete[] img1.pixels;

  libpng_image tmp2 = readPNG(mgr, normalsFile);
  imgRGB img2 = toImgRGB(tmp2);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, textures[1]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(
    GL_TEXTURE_2D, 0, GL_RGB, img2.width, img2.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img2.pixels);

  delete[] tmp2.data;
  delete[] tmp2.rowPtrs;
  delete[] img2.pixels;
}

void NormalMapModel::initPrgm(AAssetManager *mgr) {
  mProgram = glCreateProgram();
  GLuint vs = loadShader(GL_VERTEX_SHADER, getFileText(mgr, "shaders/normal_map_vs.glsl").c_str());
  GLuint fs =
    loadShader(GL_FRAGMENT_SHADER, getFileText(mgr, "shaders/normal_map_fs.glsl").c_str());
  glAttachShader(mProgram, vs);
  glAttachShader(mProgram, fs);
  glLinkProgram(mProgram);
}

void NormalMapModel::bind() {
  mMVPMatrixHandle = (GLuint) glGetUniformLocation(mProgram, "u_mvp_matrix");
  mMVMatrixHandle = (GLuint) glGetUniformLocation(mProgram, "u_mv_matrix");

  mPositionHandle = (GLuint) glGetAttribLocation(mProgram, "a_position");
  mTextCoordHandle = (GLuint) glGetAttribLocation(mProgram, "a_tex_coord");
  mNormalHandle = (GLuint) glGetAttribLocation(mProgram, "a_normal");

  mLightPosHandle = (GLuint) glGetUniformLocation(mProgram, "u_light_pos");

  mTexHandle = (GLuint) glGetUniformLocation(mProgram, "u_tex");
  mNormalMapHandle = (GLuint) glGetUniformLocation(mProgram, "u_normal_map");
}

void NormalMapModel::draw(gl_draw_info info) {
  glUseProgram(mProgram);

  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glEnableVertexAttribArray(mPositionHandle);
  glVertexAttribPointer(mPositionHandle, POSITION_SIZE, GL_FLOAT, GL_FALSE, STRIDE, 0);

  glEnableVertexAttribArray(mNormalHandle);
  glVertexAttribPointer(
    mNormalHandle, NORMAL_SIZE, GL_FLOAT, GL_FALSE, STRIDE,
    (char *) NULL + POSITION_SIZE * BYTES_PER_FLOAT);

  glEnableVertexAttribArray(mTextCoordHandle);
  glVertexAttribPointer(
    mTextCoordHandle, TEX_COORD_SIZE, GL_FLOAT, GL_FALSE, STRIDE,
    (char *) NULL + (POSITION_SIZE + NORMAL_SIZE) * BYTES_PER_FLOAT);

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glUniformMatrix4fv(mMVMatrixHandle, 1, GL_FALSE, glm::value_ptr(info.mv_matrix));

  glUniformMatrix4fv(mMVPMatrixHandle, 1, GL_FALSE, glm::value_ptr(info.mvp_matrix));

  glUniform3fv(mLightPosHandle, 1, glm::value_ptr(info.light_pos));

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures[0]);
  glUniform1i(mTexHandle, 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, textures[1]);
  glUniform1i(mNormalMapHandle, 1);

  glDrawArrays(GL_TRIANGLES, 0, nbVertex);

  glDisableVertexAttribArray(mPositionHandle);
  glDisableVertexAttribArray(mTextCoordHandle);
  glDisableVertexAttribArray(mNormalHandle);

  // checkGLError();
}

void NormalMapModel::genBuffer(AAssetManager *mgr, string objFile) {
  vector<float> packedData = parseObj(mgr, move(objFile));

  glGenBuffers(1, &buffer);

  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glBufferData(
    GL_ARRAY_BUFFER, packedData.size() * BYTES_PER_FLOAT, packedData.data(), GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

vector<float> NormalMapModel::parseObj(AAssetManager *mgr, string objFile) {
  string text = getFileText(mgr, objFile);
  vector<string> lines = split(text, '\n');

  vector<float> vertex_list;
  vector<float> normal_list;
  vector<float> uv_list;
  vector<int> vertex_draw_order;
  vector<int> normal_draw_order;
  vector<int> uv_draw_order;

  for (auto str: lines) {
    // cout << str << endl;
    vector<string> splitted_line = split(str, ' ');
    if (!splitted_line.empty()) {
      if (splitted_line[0] == "vn") {
        normal_list.push_back(stof(splitted_line[1]));
        normal_list.push_back(stof(splitted_line[2]));
        normal_list.push_back(stof(splitted_line[3]));
      } else if (splitted_line[0] == "vt") {
        uv_list.push_back(stof(splitted_line[1]));
        uv_list.push_back(stof(splitted_line[2]));
      } else if (splitted_line[0] == "v") {
        vertex_list.push_back(stof(splitted_line[1]));
        vertex_list.push_back(stof(splitted_line[2]));
        vertex_list.push_back(stof(splitted_line[3]));
      } else if (splitted_line[0] == "f") {
        vector<string> v1 = split(splitted_line[1], '/');
        vector<string> v2 = split(splitted_line[2], '/');
        vector<string> v3 = split(splitted_line[3], '/');

        vertex_draw_order.push_back(stoi(v1[0]));
        vertex_draw_order.push_back(stoi(v2[0]));
        vertex_draw_order.push_back(stoi(v3[0]));

        uv_draw_order.push_back(stoi(v1[1]));
        uv_draw_order.push_back(stoi(v2[1]));
        uv_draw_order.push_back(stoi(v3[1]));

        normal_draw_order.push_back(stoi(v1[2]));
        normal_draw_order.push_back(stoi(v2[2]));
        normal_draw_order.push_back(stoi(v3[2]));

        v1.clear();
        v2.clear();
        v3.clear();
      }
    }
    splitted_line.clear();
  }
  lines.clear();

  vector<float> packedData;
  for (int i = 0; i < vertex_draw_order.size(); i++) {
    packedData.push_back(vertex_list[(vertex_draw_order[i] - 1) * 3]);
    packedData.push_back(vertex_list[(vertex_draw_order[i] - 1) * 3 + 1]);
    packedData.push_back(vertex_list[(vertex_draw_order[i] - 1) * 3 + 2]);

    packedData.push_back(normal_list[(normal_draw_order[i] - 1) * 3]);
    packedData.push_back(normal_list[(normal_draw_order[i] - 1) * 3 + 1]);
    packedData.push_back(normal_list[(normal_draw_order[i] - 1) * 3 + 2]);

    packedData.push_back(uv_list[(uv_draw_order[i] - 1) * 2]);
    packedData.push_back(uv_list[(uv_draw_order[i] - 1) * 2 + 1]);

    nbVertex++;
  }

  vertex_list.clear();
  vertex_draw_order.clear();
  normal_list.clear();
  normal_draw_order.clear();
  uv_list.clear();
  uv_draw_order.clear();

  return packedData;
}

NormalMapModel::~NormalMapModel() {
  glDeleteTextures(2, textures);
  // TODO desaloué images textures
}
