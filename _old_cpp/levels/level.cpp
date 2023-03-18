//
// Created by samuel on 19/08/18.
//

#include "level.h"

Level::Level() : entities(), mgr(nullptr) {}

void Level::init(bool isVR, AAssetManager *mgr, btDynamicsWorld *world) {
	this->mgr = mgr;
}

AAssetManager *Level::getMgr() {
	return mgr;
}

void Level::addBases(vector<Base *> bs) {
	entities.insert(entities.end(), bs.begin(), bs.end());
}

Level::~Level() {
	entities.clear();
}