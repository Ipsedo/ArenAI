//
// Created by samuel on 30/05/18.
//

#include <android/asset_manager_jni.h>
#include <jni.h>

#include "../core/engine.h"
#include "../entity/ground/map.h"
#include "../levels/level.h"

extern "C" JNIEXPORT jlong JNICALL Java_com_samuelberrien_phyvr_MainWrappers_makeEngine(
    JNIEnv *env, jobject instance, jlong levelPtr, jobject manager) {

    Level *level = (Level *) levelPtr;

    glm::vec3 start(-1000.f, -200.f, -1000.f);
    glm::vec3 end(1000.f, 200.f, 1000.f);

    AAssetManager *mgr = AAssetManager_fromJava(env, manager);
    Engine *engine = new Engine(level, mgr);

    return (long) engine;
}

extern "C" JNIEXPORT void JNICALL Java_com_samuelberrien_phyvr_MainWrappers_updateEngine(
    JNIEnv *env, jobject instance, jlong enginePtr) {
    Engine *engine = (Engine *) enginePtr;
    engine->update(1.f / 60.f);
}

extern "C" JNIEXPORT void JNICALL Java_com_samuelberrien_phyvr_MainWrappers_freeEngine(
    JNIEnv *env, jobject instance, jlong enginePtr) {
    delete (Engine *) enginePtr;
}
