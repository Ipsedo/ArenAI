//
// Created by samuel on 30/05/18.
//

#include <jni.h>
#include "../entity/vehicles/tank.h"

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_controls_Controls_control(JNIEnv *env, jobject instance,
		jlong controlPtr, jfloat direction,
		jfloat speed, jboolean brake, jfloat turretDir, jfloat turretUp) {

	Controls *c = (Controls*) controlPtr;
	input in;
	in.xAxis = direction;
	in.speed = speed;
	in.brake = brake;
	in.turretDir = turretDir;
	in.turretUp = turretUp;
	c->onInput(in);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_getControlPtrFromCar(JNIEnv *env, jobject instance, jlong carPtr) {

	return (long) dynamic_cast <Controls*> ((Tank*)carPtr);
}

