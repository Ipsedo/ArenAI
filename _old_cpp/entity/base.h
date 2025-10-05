//
// Created by samuel on 26/06/18.
//

#ifndef PHYVR_BASETEST_H
#define PHYVR_BASETEST_H

#include "../graphics/drawable/modelvbo.h"
#include "../graphics/misc.h"
#include "btBulletDynamicsCommon.h"
#include "glm/glm.hpp"

class Base : public btRigidBody, public Drawable {
private:
    GLDrawable *drawable;
    bool hasOwnModel;

protected:
    glm::vec3 scale;

    Base(
        const btRigidBodyConstructionInfo &constructionInfo, GLDrawable *drawable,
        const glm::vec3 &scale, bool hasOwnModel);

public:
    virtual void update();

    virtual void decreaseLife(int toSub);

    virtual bool isDead();

    void draw(draw_infos infos) override;

    virtual bool needExplosion();

    virtual void onContactFinish(Base *other);

    virtual ~Base();
};

#endif// PHYVR_BASETEST_H
