//
// Created by samuel on 30/05/18.
//

#include <jni.h>
#include <android/log.h>
#include "../entity/vehicles/car.h"

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_controls_Controls_control(JNIEnv *env, jobject instance,
		jlong controlPtr, jfloat direction,
		jfloat speed, jboolean brake) {

	Controls *c = (Controls*) controlPtr;
	c->onInput(direction, speed, brake);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_getControlPtrFromCar(JNIEnv *env, jobject instance, jlong carPtr) {

	return (long) dynamic_cast <Controls*> ((Car*)carPtr);
}

