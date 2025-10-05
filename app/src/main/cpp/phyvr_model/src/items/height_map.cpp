//
// Created by samuel on 23/03/2023.
//

#include <BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h>

#include <phyvr_model/height_map.h>

HeightMapItem::HeightMapItem(
  std::string name, const std::shared_ptr<AbstractFileReader> &img_reader,
  const std::string &height_map_file, glm::vec3 pos, glm::vec3 scale)
    : Item(std::move(name)), shape_id(height_map_file), scale(scale) {
  img_rgb tmp = img_reader->read_png(height_map_file);
  img_grey img = AbstractFileReader::to_img_grey(tmp);

  auto map = new btHeightfieldTerrainShape(
    img.width, img.height, img.pixels, 1.f, 0.f, 1.f, 1, PHY_FLOAT, false);
  map->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
  map->processAllTriangles(this, btVector3(-1000., -1000., -1000.), btVector3(1000., 1000., 1000.));

  btTransform myTransform;
  myTransform.setIdentity();
  myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));

  btVector3 intertie(0.f, 0.f, 0.f);
  auto *motionState = new btDefaultMotionState(myTransform);
  btRigidBody::btRigidBodyConstructionInfo info(0.f, motionState, map, intertie);

  body = new btRigidBody(info);
}

std::shared_ptr<Shape> HeightMapItem::get_shape() {
  return std::make_shared<FromMeshShape>(shape_id, vertices, normals);
}

btRigidBody *HeightMapItem::get_body() { return body; }

glm::vec3 HeightMapItem::_get_scale() { return glm::vec3(1.); }

void HeightMapItem::processTriangle(btVector3 *triangle, int partid, int triangleindex) {
  glm::vec3 p1 = glm::vec3(triangle[0].getX(), triangle[0].getY(), triangle[0].getZ());
  glm::vec3 p2 = glm::vec3(triangle[1].getX(), triangle[1].getY(), triangle[1].getZ());
  glm::vec3 p3 = glm::vec3(triangle[2].getX(), triangle[2].getY(), triangle[2].getZ());

  glm::vec3 n1 = glm::cross(p1 - p2, p3 - p2);
  glm::vec3 n2 = glm::cross(p3 - p2, p1 - p2);

  // y : up axis
  glm::vec3 n = n1.y > 0 ? n1 : n2;

  vertices.emplace_back(p1.x, p1.y, p1.z);
  normals.emplace_back(n.x, n.y, n.z);

  vertices.emplace_back(p2.x, p2.y, p2.z);
  normals.emplace_back(n.x, n.y, n.z);

  vertices.emplace_back(p3.x, p3.y, p3.z);
  normals.emplace_back(n.x, n.y, n.z);
}
