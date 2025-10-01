//
// Created by samuel on 02/04/2023.
//

#ifndef PHYVR_CANON_H
#define PHYVR_CANON_H

#include <phyvr_controller/controller.h>
#include <phyvr_controller/inputs.h>
#include <phyvr_model/convex.h>
#include <phyvr_utils/file_reader.h>
#include <phyvr_view/camera.h>

class CanonItem : public ConvexItem,
                  public ItemProducer,
                  public Controller,
                  public Camera {
public:
  CanonItem(const std::string &prefix_name,
            const std::shared_ptr<AbstractFileReader> &file_reader,
            glm::vec3 pos, glm::vec3 rel_pos, glm::vec3 scale, float mass,
            btRigidBody *turret);

  void on_input(const user_input &input) override;

  glm::vec3 pos() override;

  glm::vec3 look() override;

  glm::vec3 up() override;

  std::vector<btTypedConstraint *> get_constraints() override;

  std::vector<std::shared_ptr<Item>> get_produced_items() override;

private:
  float angle;
  btHingeConstraint *hinge;
  std::shared_ptr<AbstractFileReader> file_reader;
  bool will_fire;
};

#endif // PHYVR_CANON_H
