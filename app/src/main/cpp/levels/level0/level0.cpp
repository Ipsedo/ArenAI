//
// Created by samuel on 29/08/18.
//

#include "level0.h"

void Level0::init(bool isVR, AAssetManager *mgr, btDynamicsWorld *world) {
	Level::init(isVR, mgr, world);
}

vector<Controls *> Level0::getControls() {
	return vector<Controls *>();
}

Camera *Level0::getCamera() {
	return nullptr;
}

vector<Shooter *> Level0::getShooters() {
	return vector<Shooter *>();
}

vector<Base *> Level0::getEntities() {
	return vector<Base *>();
}

vector<Drawable *> Level0::getDrawables() {
	return vector<Drawable *>();
}

Limits *Level0::getLimits() {
	return nullptr;
}

bool Level0::won() {
	return false;
}

bool Level0::lose() {
	return false;
}

Level0::~Level0() {

}
