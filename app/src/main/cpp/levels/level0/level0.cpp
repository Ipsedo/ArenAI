//
// Created by samuel on 11/06/18.
//
#include <thread>

#include "level0.h"


Level0::Level0() {
	isInit = false;
}

void Level0::init() {
	isInit = true;
}



vector<Base *> Level0::getEntities() {
	return entities;
}

bool Level0::won() {
	return false;
}

bool Level0::lose() {
	return false;
}

vector<Shooter *> Level0::getShooters() {
	vector<Shooter*> res = vector<Shooter*>();
	res.push_back(tank);
	return res;
}

Camera *Level0::getCamera() {
	return (Camera*) tank;
}

Controls *Level0::getControls() {
	return (Controls*) tank;
}
