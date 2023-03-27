//
// Created by samuel on 23/03/2023.
//

#ifndef PHYVR_HEIGHT_MAP_H
#define PHYVR_HEIGHT_MAP_H

#include <btBulletDynamicsCommon.h>

#include "../item.h"

class HeightMapItem : public Item, public btTriangleCallback {
public:
  HeightMapItem(std::string name, AAssetManager *mgr,
                const std::string &height_map_file, glm::vec3 pos,
                glm::vec3 scale);

  std::shared_ptr<Shape> get_shape() override;

  btRigidBody *get_body() override;

  void processTriangle(btVector3 *triangle, int partid,
                       int triangleindex) override;

  nlohmann::json get_state() override;

  void from_state(const nlohmann::json &state) override;

protected:
  glm::vec3 _get_scale() override;

private:
  std::shared_ptr<Shape> shape;
  glm::vec3 scale;

  std::vector<std::tuple<float, float, float>> vertices;
  std::vector<std::tuple<float, float, float>> normals;

  btRigidBody *body;
};

#endif // PHYVR_HEIGHT_MAP_H
