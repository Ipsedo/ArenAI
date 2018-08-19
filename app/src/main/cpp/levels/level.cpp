//
// Created by samuel on 19/08/18.
//

#include "level.h"

void Level::addBases(vector<Base *> bs) {
	entities.insert(entities.end(), bs.begin(), bs.end());
}

Level::~Level() {}