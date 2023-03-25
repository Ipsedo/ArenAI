//
// Created by samuel on 23/03/2023.
//

#ifndef PHYVR_CONVEX_H
#define PHYVR_CONVEX_H

#include "../item.h"

class ConvexItem : public Item {
public:
  ConvexItem(std::string name, const std::shared_ptr<Shape> &shape,
             glm::vec3 position, glm::vec3 scale, float mass);

  std::shared_ptr<Shape> get_shape() override;

  btRigidBody *get_body() override;

protected:
  glm::vec3 _get_scale() override;

private:
private:
  std::string name;

  std::shared_ptr<Shape> shape;

  btRigidBody *body;
  btCollisionShape *collision_shape;

  glm::vec3 scale;
};

#endif // PHYVR_CONVEX_H
