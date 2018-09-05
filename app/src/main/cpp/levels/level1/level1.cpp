//
// Created by samuel on 29/08/18.
//

#include "level1.h"
#include "../../entity/vehicles/tank/tank.h"
#include "../../entity/ground/map.h"

Level1::Level1() : isInit(false), cibles(vector<Cible *>()) {}

void Level1::init(bool isVR, AAssetManager *mgr, btDynamicsWorld *world) {
	Level::init(isVR, mgr, world);
	tank = new Tank(isVR, mgr, world, btVector3(5.f, -35.f, 20.f));
	vector<Base *> tankBases = tank->getBase();
	entities.insert(entities.end(), tankBases.begin(), tankBases.end());

	libpng_image tmp = readPNG(mgr, "heightmap/heightmap6.png");
	normalized_image img = toGrayImg(tmp);
	float *array = new float[img.allpixels.size()];
	for (int i = 0; i < img.allpixels.size(); i++) {
		array[i] = img.allpixels[i];
	}
	Map *sol = new Map(array, img.width, img.height, btVector3(0.f, 40.f, 0.f), btVector3(10.f, 200.f, 10.f));
	entities.push_back(sol);

	SupportCible *supportCible = new SupportCible(mgr, glm::vec3(5.f, -40.f, 100.f));
	Cible *cible = new Cible(mgr, supportCible, world);
	entities.push_back(supportCible);
	entities.push_back(cible);

	cibles.push_back(cible);

	isInit = true;
}

vector<Controls *> Level1::getControls() {
	if (!isInit)
		return vector<Controls *>();
	return tank->getControls();
}

Camera *Level1::getCamera() {
	if (!isInit)
		return nullptr;
	return tank->getCamera();
}

vector<Shooter *> Level1::getShooters() {
	if (!isInit)
		return vector<Shooter *>();
	return tank->getShooters();
}

vector<Base *> Level1::getEntities() {
	if (!isInit)
		return vector<Base *>();
	return entities;
}

vector<Drawable *> Level1::getDrawables() {
	if (!isInit)
		return vector<Drawable *>();

	vector<Drawable *> d;
	for (Base *b : entities)
		d.push_back(b);
	return d;
}

Limits Level1::getLimits() {
	glm::vec3 start(-1000.f, -200.f, -1000.f);
	glm::vec3 end(1000.f, 200.f, 1000.f);
	return BoxLimits(start, end - start);
}

bool Level1::won() {
	bool won = true;
	for (Cible *c : cibles)
		won &= c->isWon();
	return won;
}

bool Level1::lose() {
	return false;
}

void Level1::step() {

}

Level1::~Level1() {

}
