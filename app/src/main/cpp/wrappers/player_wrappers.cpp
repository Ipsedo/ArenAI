//
// Created by samuel on 30/05/18.
//


#include <jni.h>
#include <android/asset_manager_jni.h>
#include "entity/vehicles/tank/tank.h"
#include "../core/engine.h"
#include <android/log.h>

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

	/*libpng_image libpngImage = readPNG(cppMgr, "image/canard.png");
	colored_image coloredImage = toRGBImg(libpngImage);

	for (int i = 0; i < coloredImage.height; i++) {
		string row = "";
		for (int j = 0; j < coloredImage.width; j++) {
			int idx = i * coloredImage.width + j;
			color<int> c = coloredImage.allpixels[idx];
			row += float(c.r + c.g + c.b) / (3.f * coloredImage.maxValue) > 0.5f ? " #" : " .";
		}
		__android_log_print(ANDROID_LOG_DEBUG, "PhyVR", "%s", row.c_str());
	}*/
	return (long) tank;
}
