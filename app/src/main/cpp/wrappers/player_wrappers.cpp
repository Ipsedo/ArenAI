//
// Created by samuel on 30/05/18.
//


#include <jni.h>
#include <android/asset_manager_jni.h>
#include "entity/vehicles/tank/tank.h"
#include "../core/engine.h"
#include "../controls/controls.h"

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_wrappers_MainWrappers_initPlayer(JNIEnv *env, jobject instance, jobject assetManager,
															  jlong enginePtr, jlong rendererPtr, jlong entityPtr,
															  jboolean vr) {

	Engine *level = (Engine *) enginePtr;
	Renderer *renderer = (Renderer *) rendererPtr;
	vector<Base *> *entity = (vector<Base *> *) entityPtr;
	AAssetManager *cppMgr = AAssetManager_fromJava(env, assetManager);

	Player *tank = new Tank(vr, cppMgr, level->world, btVector3(5.f, -10.f, 20.f));
	for (Base *b : tank->getBase())
		entity->push_back(b);
	renderer->setCamera(tank->getCamera());
	for (Shooter *s : tank->getShooters())
		level->addShooter(s);
	return (long) tank;
}
