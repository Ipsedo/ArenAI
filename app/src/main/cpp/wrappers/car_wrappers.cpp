//
// Created by samuel on 30/05/18.
//


#include <jni.h>
#include <android/asset_manager_jni.h>
#include "../entity/vehicles/tank.h"
#include "../level/level.h"
#include "../controls/controls.h"

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_initCar(JNIEnv *env, jobject instance, jobject assetManager,
											   jlong levelPtr, jlong rendererPtr, jlong entityPtr) {

	Level *level = (Level*) levelPtr;
	Renderer *renderer = (Renderer*) rendererPtr;
	vector<Base*>* entity = (vector<Base*>*) entityPtr;
	AAssetManager *cppMgr = AAssetManager_fromJava(env, assetManager);

	Tank *c = new Tank(glm::vec3(0, -15, -20), level->world, cppMgr, entity);
	level->addShooter(c);
	entity->push_back(c);
	renderer->setCamera(c);

	return (long) c;
}
