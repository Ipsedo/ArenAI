//
// Created by samuel on 13/08/18.
//

#include "turret.h"
#include <glm/glm.hpp>
#include "../../graphics/drawable/normalmodel.h"
#include "../../utils/rigidbody.h"
#include "../../utils/assets.h"
#include "../../utils/vec.h"

/////////////////////
// Turret
/////////////////////
NormalMapModel *makeTurretModel(AAssetManager *mgr) {
	return new NormalMapModel(mgr, "obj/tank_turret.obj", "textures/turret_tex.png", "textures/199_norm.png");
}

Turret::Turret(AAssetManager *mgr, btDynamicsWorld *world, Base *chassis, btVector3 chassisPos)
		: Poly([mgr](glm::vec3 scale) {
				   string turretObjTxt = getFileText(mgr, "obj/tank_turret.obj");
				   btCollisionShape *turretShape = parseObj(turretObjTxt);
				   turretShape->setLocalScaling(btVector3(turretScale.x, turretScale.y, turretScale.z));
				   return turretShape;
			   },
			   makeTurretModel(mgr),
			   btVector3ToVec3(chassisPos + turretRelPos), turretScale, glm::mat4(1.0f), turretMass, true),
		  angle(0.f), respawn(false), added(0.f), pos(chassisPos + turretRelPos) {
	btTransform tr;
	tr.setIdentity();
	tr.setOrigin(pos);

	btRigidBody *pBodyA = chassis;
	btRigidBody *pBodyB = this;

	btVector3 pivotA = btVector3(0.f, chassisScale.y, 0.f);
	btVector3 pivotB = btVector3(0.f, -turretScale.y, 0.f);
	btVector3 axis = btVector3(0.f, 1.f, 0.f);

	hinge = new btHingeConstraint(*pBodyA, *pBodyB, pivotA, pivotB, axis, axis, true);
	world->addConstraint(hinge, true);
	hinge->setLimit(0, 0);
}

void Turret::onInput(input in) {
	added = in.turretDir;
	respawn = in.respawn;
}

void Turret::update() {
	Base::update();

	angle += added * 3e-2f;
	angle = angle > 1.f ? 1.f : angle;
	angle = angle < -1.f ? -1.f : angle;
	hinge->setLimit(angle * (float) M_PI * 0.6f, angle * (float) M_PI * 0.6f);

	if (respawn) {
		btTransform tr;
		tr.setIdentity();
		tr.setOrigin(pos);
		getMotionState()->setWorldTransform(tr);
		setWorldTransform(tr);
		clearForces();
		setLinearVelocity(btVector3(0, 0, 0));
		setAngularVelocity(btVector3(0, 0, 0));
		respawn = false;
	}
}