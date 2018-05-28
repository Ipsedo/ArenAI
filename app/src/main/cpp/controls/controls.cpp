//
// Created by samuel on 28/05/18.
//

#include "controls.h"
#include <jni.h>

std::tuple<int, int> Controls::getJoystick1() {
    return std::tuple<int, int>();
}

std::tuple<int, int> Controls::getJoystick2() {
    return std::tuple<int, int>();
}

bool Controls::isPressingFire() {
    return false;
}

bool Controls::isPressingBrake() {
    return false;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_controls_Controls_initControls(JNIEnv *env, jobject instance) {
    return (long) new Controls();
}
