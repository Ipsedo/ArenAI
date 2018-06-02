#include <jni.h>
#include <string>
#include <btBulletDynamicsCommon.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <glm/gtc/type_ptr.hpp>

#include "../utils/assets.h"
#include "../graphics/renderer.h"
#include "../core/engine.h"
#include "../entity/convex.h"
#include "../entity/ground/map.h"
#include "../entity/vehicles/car2.h"
#include "../entity/poly/cone.h"
#include "../entity/poly/box.h"
#include "../entity/poly/sphere.h"
#include "../entity/poly/cylinder.h"
#include "../entity/vehicles/car.h"
#include "wrapper_utils.h"

#define HEIGHT_SPAWN 30.f

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_initEntity(JNIEnv *env, jobject instance,
												  jobject assetManager, jfloatArray heightmap_,
												  jint width, jint height) {
	jfloat *heightmap = env->GetFloatArrayElements(heightmap_, NULL);
	float *map = jfloatPtrToCppFloatPtr(heightmap, width * height);

	AAssetManager *cppMgr = AAssetManager_fromJava(env, assetManager);

	glm::mat4 id(1.f);

	Base *sol = new Map(glm::vec3(0.f, 0.f, 0.f), width, height, map, glm::vec3(10.f, 50.f, 10.f));
	//new Box(cppMgr, glm::vec3(0.f, -5.f, 0.f), glm::vec3(40.f,0.1f,40.f), id, 0.f);

	vector<Base *> *boxes = new vector<Base *>();
	boxes->push_back(sol);

	int nbEntity = 10;
	float maxMass = 100.f;
	for (int i = 0; i < nbEntity; i++) {
		float x = 40.f * (float) rand() / RAND_MAX - 20.f;
		float z = 40.f * (float) rand() / RAND_MAX - 20.f;
		float scale = 2.f * (float) rand() / RAND_MAX;
		float mass = maxMass * float(rand()) / RAND_MAX;
		boxes->push_back(
				new Box(cppMgr,
							 glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale),
							 id, mass));
	}
	for (int i = 0; i < nbEntity; i++) {
		float x = 40.f * (float) rand() / RAND_MAX - 20.f;
		float z = 40.f * (float) rand() / RAND_MAX - 20.f;
		float scale = 2.f * (float) rand() / RAND_MAX;
		float mass = maxMass * float(rand()) / RAND_MAX;
		boxes->push_back(
				new Cylinder(cppMgr,
							 glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale),
							 id, mass));
	}
	for (int i = 0; i < nbEntity; i++) {
		float x = 40.f * (float) rand() / RAND_MAX - 20.f;
		float z = 40.f * (float) rand() / RAND_MAX - 20.f;
		float scale = 2.f * (float) rand() / RAND_MAX;
		float mass = maxMass * float(rand()) / RAND_MAX;
		boxes->push_back(
				new Cone(cppMgr,
							 glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale),
							 id, mass));
	}
	for (int i = 0; i < nbEntity; i++) {
		float x = 40.f * (float) rand() / RAND_MAX - 20.f;
		float z = 40.f * (float) rand() / RAND_MAX - 20.f;
		float scale = 2.f * (float) rand() / RAND_MAX;
		float mass = maxMass * float(rand()) / RAND_MAX;
		boxes->push_back(
				new Sphere(cppMgr,
							 glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale),
							 id, mass));
	}
	return (long) boxes;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_addBox(JNIEnv *env, jobject instance, jobject assetManager,
											  jlong boxesPtr) {

	vector<Base *> *boxes = (vector<Base *> *) boxesPtr;

	AAssetManager *cppMgr = AAssetManager_fromJava(env, assetManager);

	glm::mat4 id(1.f);
	double r = (double) rand() / RAND_MAX;

	if (r < 1. / 60.) {
		float x = 40.f * (float) rand() / RAND_MAX - 20.f;
		float z = 40.f * (float) rand() / RAND_MAX - 20.f;
		float scale = 2.f * (float) rand() / RAND_MAX;
		float mass = scale;
		Base *base;
		if ((float) rand() / RAND_MAX > 0.8) {
			base = new Cylinder(cppMgr,
								glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale),
								id, mass);
		} else {
			/*std::string m =
					(float) rand() / RAND_MAX > 0.5 ? "obj/icosahedron.obj" : "obj/ast1.obj";*/
			std::string m = "obj/icosahedron.obj";
			base = new Convex(cppMgr, m,
							  glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale),
							  id, mass);
		}
		boxes->push_back(base);
	}

}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_freeBoxes(JNIEnv *env, jobject instance, jlong boxesPtr) {

	vector<Base *> *boxes = (vector<Base *> *) boxesPtr;
	for (Base *b : *boxes)
		delete b;
	boxes->clear();
	delete boxes;

}

