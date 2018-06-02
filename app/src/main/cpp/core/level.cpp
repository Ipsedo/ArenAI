//
// Created by samuel on 02/06/18.
//

#include "level.h"

Level::Level() {
	isInit = false;
}

void Level::init() {
	isInit = true;
}

Player* Level::getPlayer() {
	return isInit ? player : nullptr;
}

vector<Base*>* Level::getEntities() {
	return isInit ? entities : nullptr;
}
