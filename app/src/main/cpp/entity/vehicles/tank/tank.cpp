//
// Created by samuel on 26/06/18.
//

#include "tank.h"
#include "chassis.h"

Tank::Tank(bool vr, AAssetManager *mgr, btDynamicsWorld *world, btVector3 centerPos) {
	chassis = makeChassis(mgr, centerPos);
	chassis->setActivationState(DISABLE_DEACTIVATION);

	wheels.push_back(makeFrontWheel(mgr, world, chassis, centerPos, wheelPos[0]));
	wheels.push_back(makeFrontWheel(mgr, world, chassis, centerPos, wheelPos[1]));

	wheels.push_back(makeWheel(mgr, world, chassis, centerPos, wheelPos[2]));
	wheels.push_back(makeWheel(mgr, world, chassis, centerPos, wheelPos[3]));
	wheels.push_back(makeWheel(mgr, world, chassis, centerPos, wheelPos[4]));
	wheels.push_back(makeWheel(mgr, world, chassis, centerPos, wheelPos[5]));

	turret = makeTurret(mgr, world, chassis, centerPos);

	canon = makeCanon(mgr, world, turret, centerPos + turretRelPos);

	for (btRigidBody* rb : wheels) canon->setIgnoreCollisionCheck(rb, true);
	canon->setIgnoreCollisionCheck(chassis, true);
	canon->setIgnoreCollisionCheck(turret, true);

	if (vr) camera = chassis;
	else camera = canon;
	//camera = chassis;
}

vector<Base *> Tank::getBaseTest() {
	vector<Base *> res;
	res.push_back(chassis);
	res.push_back(turret);
	res.push_back(canon);
	for (Base *b : wheels)
		res.push_back(b);
	return res;
}

vector<Controls *> Tank::getControls() {
	vector<Controls *> res;
	res.push_back(turret);
	res.push_back(canon);
	for (Controls *c : wheels)
		res.push_back(c);
	return res;
}

Camera *Tank::getCamera() {
	return camera;
}

vector<Shooter *> Tank::getShooters() {
	vector<Shooter *> res;
	res.push_back(canon);
	return res;
}
