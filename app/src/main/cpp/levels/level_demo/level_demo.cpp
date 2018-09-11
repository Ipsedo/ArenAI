//
// Created by samuel on 11/06/18.
//
#include "level_demo.h"
#include "../../entity/ground/map.h"
#include "../../entity/simple/box.h"
#include "../../entity/simple/cylinder.h"
#include "../../entity/simple/sphere.h"
#include "../../entity/simple/cone.h"

#define HEIGHT_SPAWN 30.f

LevelDemo::LevelDemo() : isInit(false) {}

void LevelDemo::init(bool isVR, AAssetManager *mgr, btDynamicsWorld *world) {
	tank = new Tank(isVR, mgr, world, btVector3(5.f, -35.f, 20.f));
	vector<Base *> tankBases = tank->getBase();
	entities.insert(entities.end(), tankBases.begin(), tankBases.end());

	// items
	libpng_image tmp = readPNG(mgr, "heightmap/heightmap6.png");
	normalized_image img = toGrayImg(tmp);
	float *array = new float[img.allpixels.size()];
	for (int i = 0; i < img.allpixels.size(); i++) {
		array[i] = img.allpixels[i];
	}
	delete[] tmp.rowPtrs;
	delete[] tmp.data;
	img.allpixels.clear();
	Map *sol = new Map(array, img.width, img.height, btVector3(0.f, 40.f, 0.f), btVector3(10.f, 200.f, 10.f));
	entities.push_back(sol);

	int width = 100, height = 100;

	glm::mat4 id(1.f);

	int nbEntity = 100;
	float spawnRange = 10.f * min(width, height) * 0.5f;
	float maxMass = 100.f;
	for (int i = 0; i < nbEntity; i++) {
		float x = spawnRange * (float) rand() / RAND_MAX - spawnRange * 0.5f;
		float z = spawnRange * (float) rand() / RAND_MAX - spawnRange * 0.5f;
		float scale = 2.f * (float) rand() / RAND_MAX;
		float mass = maxMass * float(rand()) / RAND_MAX;
		entities.push_back(new Box(mgr, glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale), id, mass));
	}
	for (int i = 0; i < nbEntity; i++) {
		float x = spawnRange * (float) rand() / RAND_MAX - spawnRange * 0.5f;
		float z = spawnRange * (float) rand() / RAND_MAX - spawnRange * 0.5f;
		float scale = 2.f * (float) rand() / RAND_MAX;
		float mass = maxMass * float(rand()) / RAND_MAX;

		entities.push_back(new Cylinder(mgr, glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale), id, mass));
	}
	for (int i = 0; i < nbEntity; i++) {
		float x = spawnRange * (float) rand() / RAND_MAX - spawnRange * 0.5f;
		float z = spawnRange * (float) rand() / RAND_MAX - spawnRange * 0.5f;
		float scale = 2.f * (float) rand() / RAND_MAX;
		float mass = maxMass * float(rand()) / RAND_MAX;

		entities.push_back(new Cone(mgr, glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale), id, mass));
	}
	for (int i = 0; i < nbEntity; i++) {
		float x = spawnRange * (float) rand() / RAND_MAX - spawnRange * 0.5f;
		float z = spawnRange * (float) rand() / RAND_MAX - spawnRange * 0.5f;
		float scale = 2.f * (float) rand() / RAND_MAX;
		float mass = maxMass * float(rand()) / RAND_MAX;
		entities.push_back(new Sphere(mgr, glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale), id, mass));
	}

	map = new CubeMap(mgr, "cubemap/2/", 1000.f);

	isInit = true;
}


vector<Base *> LevelDemo::getEntities() {
	if (!isInit)
		exit(666);

	return entities;
}

bool LevelDemo::won() {
	return false;
}

bool LevelDemo::lose() {
	return false;
}

vector<Shooter *> LevelDemo::getShooters() {
	if (!isInit)
		exit(666);

	return tank->getShooters();
}

Camera *LevelDemo::getCamera() {
	if (!isInit)
		exit(666);

	return tank->getCamera();
}

vector<Controls *> LevelDemo::getControls() {
	if (!isInit)
		exit(666);

	return tank->getControls();
}

vector<Drawable *> LevelDemo::getDrawables() {
	if (!isInit)
		exit(666);

	vector<Drawable *> d;
	for (Base *b : entities)
		d.push_back(b);
	d.push_back(map);
	return d;
}

Limits LevelDemo::getLimits() {
	glm::vec3 start(-1000.f, -200.f, -1000.f);
	glm::vec3 end(1000.f, 200.f, 1000.f);
	return BoxLimits(start, end - start);
}

void LevelDemo::step() {

}

LevelDemo::~LevelDemo() {
	delete map;
	delete tank;
}
