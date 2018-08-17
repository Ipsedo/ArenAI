//
// Created by samuel on 30/05/18.
//

#include <jni.h>
#include "entity/vehicles/tank/tank.h"
#include "wrapper_utils.h"

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_controls_Controls_control(JNIEnv *env, jobject instance,
													   jlong controlPtr, jfloat direction,
													   jfloat speed, jboolean brake, jfloat turretDir, jfloat turretUp,
													   jboolean respawn, jboolean fire) {

	vector<Controls *> *ctrl = (vector<Controls *> *) controlPtr;
	input in;
	in.xAxis = direction;
	in.speed = speed;
	in.brake = brake;
	in.turretDir = -turretDir;
	in.turretUp = turretUp;
	in.respawn = respawn;
	in.fire = fire;
	for (Controls *c : *ctrl)
		c->onInput(in);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_wrappers_MainWrappers_getControlPtrFromPlayer(JNIEnv *env, jobject instance,
																		   jlong carPtr) {

	vector<Controls *> *res = new vector<Controls *>();
	for (Controls *c : ((Player *) carPtr)->getControls())
		res->push_back(c);
	return (long) res;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_controls_Controls_control2(JNIEnv *env, jobject instance,
														jlong controlPtr, jfloatArray arrayControl_) {
	jfloat *arrayControl = env->GetFloatArrayElements(arrayControl_, NULL);

	float *controls = jfloatPtrToCppFloatPtr(arrayControl, 8);
	vector<Controls *> *ctrl = (vector<Controls *> *) controlPtr;
	input in;
	in.xAxis = controls[0];
	in.speed = controls[1];
	in.brake = controls[2] != 0.f;
	in.turretDir = controls[3];
	in.turretUp = controls[4];
	in.respawn = controls[5] != 0.f;
	in.fire = controls[6] != 0.f;
	for (Controls *c : *ctrl)
		c->onInput(in);

	delete controls;
	env->ReleaseFloatArrayElements(arrayControl_, arrayControl, 0);
}

