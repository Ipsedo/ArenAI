//
// Created by samuel on 19/03/2023.
//

#ifndef PHYVR_CUBEMAP_H
#define PHYVR_CUBEMAP_H

#include "./drawable.h"

#include <string>

#include "../constants.h"
#include "../program.h"

class CubeMap : public Drawable {
private:
  static const int POSITION_SIZE = 3;
  static const int STRIDE = POSITION_SIZE * BYTES_PER_FLOAT;

  std::unique_ptr<Program> program;

  int nb_vertices;

public:
  CubeMap(AAssetManager *mgr, const std::string &pngs_root_path);

  void draw(glm::mat4 mvp_matrix, glm::mat4 mv_matrix,
            glm::vec3 light_pos_from_camera, glm::vec3 camera_pos) override;
};

#endif // PHYVR_CUBEMAP_H
