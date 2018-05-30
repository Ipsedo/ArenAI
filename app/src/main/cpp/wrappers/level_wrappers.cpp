//
// Created by samuel on 30/05/18.
//

#include <jni.h>
#include "../level/level.h"

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_initLevel(JNIEnv *env, jobject instance, jlong boxesPtr) {

	vector<Base *> *boxes = (vector<Base *> *) boxesPtr;

	Level *level = new Level(boxes);

	return (long) level;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_updateLevel(JNIEnv *env, jobject instance, jlong levelptr) {
	Level *level = (Level *) levelptr;
	level->update(1.f / 60.f);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_freeLevel(JNIEnv *env, jobject instance, jlong levelPtr) {
	delete (Level *) levelPtr;
}