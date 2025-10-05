//
// Created by samuel on 30/05/18.
//

#import <numeric>

#include <jni.h>

#include "../entity/tank/tank.h"
#include "../levels/level.h"
#include "wrapper_utils.h"

extern "C" JNIEXPORT void JNICALL Java_com_samuelberrien_phyvr_controls_GamePad_control(
    JNIEnv *env, jobject instance, jlong levelPtr, jfloat direction, jfloat speed, jboolean brake,
    jfloat turretDir, jfloat turretUp, jboolean respawn, jboolean fire) {

    vector<Controls *> ctrl = ((Level *) levelPtr)->getControls();
    input in;
    in.xAxis = (float) direction == NULL ? 0.f : direction;
    in.speed = (float) speed == NULL ? 0.f : speed;
    in.brake = (bool) brake == NULL ? false : brake;
    in.turretDir = (float) -turretDir == NULL ? 0.f : -turretDir;
    in.turretUp = (float) turretUp == NULL ? 0.f : turretUp;
    in.respawn = (bool) respawn == NULL ? false : respawn;
    in.fire = (bool) fire == NULL ? false : fire;
    for (Controls *c: ctrl) c->onInput(in);
    ctrl.clear();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_samuelberrien_phyvr_controls_GamePad_vibrate(JNIEnv *env, jobject thiz, jlong level_ptr) {
    vector<Controls *> ctrl = ((Level *) level_ptr)->getControls();

    bool vibrate = false;
    for (Controls *c: ctrl) vibrate |= c->getOutput().vibrate;
    return (jboolean) vibrate;
}

extern "C" JNIEXPORT void JNICALL Java_com_samuelberrien_phyvr_controls_GamePad_control2(
    JNIEnv *env, jobject instance, jlong levelPtr, jfloatArray arrayControl_) {
    jfloat *arrayControl = env->GetFloatArrayElements(arrayControl_, NULL);

    float *controls = jfloatPtrToCppFloatPtr(arrayControl, 8);
    vector<Controls *> ctrl = ((Level *) levelPtr)->getControls();
    input in;
    in.xAxis = controls[0];
    in.speed = controls[1];
    in.brake = controls[2] != 0.f;
    in.turretDir = controls[3];
    in.turretUp = controls[4];
    in.respawn = controls[5] != 0.f;
    in.fire = controls[6] != 0.f;
    for (Controls *c: ctrl) c->onInput(in);

    delete[] controls;
    env->ReleaseFloatArrayElements(arrayControl_, arrayControl, 0);
    ctrl.clear();
}

extern "C" JNIEXPORT void JNICALL Java_com_samuelberrien_phyvr_controls_UI_control(
    JNIEnv *env, jobject instance, jlong levelPtr, jfloat direction, jfloat speed, jboolean brake,
    jfloat turretDir, jfloat turretUp, jboolean respawn, jboolean fire) {

    vector<Controls *> ctrl = ((Level *) levelPtr)->getControls();
    input in;
    in.xAxis = (float) direction == NULL ? 0.f : direction;
    in.speed = (float) speed == NULL ? 0.f : speed;
    in.brake = (bool) brake == NULL ? false : brake;
    in.turretDir = (float) -turretDir == NULL ? 0.f : -turretDir;
    in.turretUp = (float) turretUp == NULL ? 0.f : turretUp;
    in.respawn = (bool) respawn == NULL ? false : respawn;
    in.fire = (bool) fire == NULL ? false : fire;
    for (Controls *c: ctrl) c->onInput(in);
    ctrl.clear();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_samuelberrien_phyvr_controls_UI_vibrate(JNIEnv *env, jobject thiz, jlong level_ptr) {
    vector<Controls *> ctrl = ((Level *) level_ptr)->getControls();

    bool vibrate = false;
    for (Controls *c: ctrl) vibrate |= c->getOutput().vibrate;
    return (jboolean) vibrate;
}
