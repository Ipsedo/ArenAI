//
// Created by samuel on 29/08/18.
//

#include "level1.h"
#include "../../entity/tank/tank.h"
#include "../../entity/ground/map.h"

Level1::Level1() : isInit(false), cibles(vector<Cible *>()), compass(vector<Compass *>()) {}

void Level1::init(bool isVR, AAssetManager *mgr, btDynamicsWorld *world) {
	Level::init(isVR, mgr, world);
	tank = new Tank(isVR, mgr, world, btVector3(-97.f, 1.f, -175.f));
	vector<Base *> tankBases = tank->getBases();
	entities.insert(entities.end(), tankBases.begin(), tankBases.end());

	libpng_image tmp = readPNG(mgr, "heightmap/heightmap_road2.png");
	normalized_image img = toGrayImg(tmp);
	float *array = new float[img.allpixels.size()];
	for (int i = 0; i < img.allpixels.size(); i++) {
		array[i] = img.allpixels[i];
	}
	delete[] tmp.rowPtrs;
	delete[] tmp.data;
	img.allpixels.clear();
	Map *sol = new Map(array, img.width, img.height, btVector3(0.f, 0.f, 0.f), btVector3(0.8f, 20.f, 0.8f));
	entities.push_back(sol);

	SupportCible *supportCible1 = new SupportCible(mgr, glm::vec3(-28.f, 15.f, 40.f));
	Cible *cible1 = new Cible(mgr, supportCible1, world);
	entities.push_back(supportCible1);
	entities.push_back(cible1);
	cibles.push_back(cible1);
	compass.push_back(new Compass(cible1));

	SupportCible *supportCible2 = new SupportCible(mgr, glm::vec3(-77.f, 15.f, 150.f));
	Cible *cible2 = new Cible(mgr, supportCible2, world);
	entities.push_back(supportCible2);
	entities.push_back(cible2);
	cibles.push_back(cible2);
	compass.push_back(new Compass(cible2));

	SupportCible *supportCible3 = new SupportCible(mgr, glm::vec3(73.f, 15.f, 40.f));
	Cible *cible3 = new Cible(mgr, supportCible3, world);
	entities.push_back(supportCible3);
	entities.push_back(cible3);
	cibles.push_back(cible3);
	compass.push_back(new Compass(cible3));

	map = new CubeMap(mgr, "cubemap/1/", 1000.f);

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
	for (int i = 0; i < compass.size(); i++)
		if (!cibles[i]->isWon())
			d.push_back(compass[i]);
	d.push_back(map);
	for (Drawable *dr : tank->getDrawables())
		d.push_back(dr);
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
	delete map;
	delete tank;
}

glm::vec3 Level1::getLightPos() {
	return glm::vec3(0,200,0);
}
