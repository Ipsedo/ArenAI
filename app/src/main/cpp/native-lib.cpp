#include <jni.h>
#include <string>
#include <btBulletDynamicsCommon.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "utils/assets.h"
#include "level/level.h"
#include <glm/gtc/type_ptr.hpp>

extern "C" JNIEXPORT jstring

JNICALL
Java_com_samuelberrien_phyvr_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {

    // Build the broadphase
    btBroadphaseInterface* broadphase = new btDbvtBroadphase();

    // Set up the collision configuration and dispatcher
    btDefaultCollisionConfiguration* collisionConfiguration = new btDefaultCollisionConfiguration();
    btCollisionDispatcher* dispatcher = new btCollisionDispatcher(collisionConfiguration);

    // The actual physics solver
    btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

    // The world.
    btDiscreteDynamicsWorld* dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);
    dynamicsWorld->setGravity(btVector3(0, -10, 0));
    float tmpX = float(dynamicsWorld->getGravity().getX());
    float tmpY = float(dynamicsWorld->getGravity().getY());
    float tmpZ = float(dynamicsWorld->getGravity().getZ());

    delete dynamicsWorld;
    delete solver;
    delete dispatcher;
    delete collisionConfiguration;
    delete broadphase;

    char hello[100];
    sprintf(hello, "Hello\nGravity x = %f, y = %f, z = %f", tmpX, tmpY, tmpZ);
    return env->NewStringUTF(hello);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_samuelberrien_phyvr_MainActivity_test(JNIEnv *env, jobject instance, jobject mgr) {

    AAssetManager* cppMgr = AAssetManager_fromJava(env, mgr);

    std::string objTxt = getFileText(cppMgr, "obj/icosahedron.obj");
    return env->NewStringUTF(objTxt.c_str());
}

/**
 * Level stuff
 */

float* jfloatPtrToCppFloatPtr(jfloat* array, int length) {
    float* res = new float[length];
    for (int i = 0; i < length; i++) {
        res[i] = array[i];
    }
    return res;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_initLevel(JNIEnv *env, jobject instance,
                                                 jobject assetManager) {
    AAssetManager* cppMgr = AAssetManager_fromJava(env, assetManager);
    Level* level = new Level(cppMgr);
    level->init();
    return (long) level;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_updateLevel(JNIEnv *env, jobject instance, jlong levelptr,
                                                   jfloatArray mHeadView_) {
    jfloat *mHeadView = env->GetFloatArrayElements(mHeadView_, NULL);
    float *headView = jfloatPtrToCppFloatPtr(mHeadView, 16);

    Level* level = (Level*) levelptr;
    level->update(glm::make_mat4(headView));

    env->ReleaseFloatArrayElements(mHeadView_, mHeadView, 0);
    delete headView;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_drawLevel(JNIEnv *env, jobject instance, jlong levelptr,
                                                 jfloatArray mEyeProjectionMatrix_,
                                                 jfloatArray mEyeViewMatrix_,
                                                 jfloatArray myLighPosInEyeSpace_,
                                                 jfloatArray mCameraPos_) {
    jfloat *mEyeProjectionMatrix = env->GetFloatArrayElements(mEyeProjectionMatrix_, NULL);
    jfloat *mEyeViewMatrix = env->GetFloatArrayElements(mEyeViewMatrix_, NULL);
    jfloat *myLighPosInEyeSpace = env->GetFloatArrayElements(myLighPosInEyeSpace_, NULL);
    jfloat *mCameraPos = env->GetFloatArrayElements(mCameraPos_, NULL);

    float* eyeProjectionMatrix = jfloatPtrToCppFloatPtr(mEyeProjectionMatrix, 16);
    float* eyeViewMatrix = jfloatPtrToCppFloatPtr(mEyeViewMatrix, 16);
    float* lighPosInEyeSpace = jfloatPtrToCppFloatPtr(myLighPosInEyeSpace, 4);
    float* cameraPos = jfloatPtrToCppFloatPtr(mCameraPos, 3);

    Level* level = (Level*) levelptr;
    level->draw(glm::make_mat4(eyeProjectionMatrix),
                glm::make_mat4(eyeViewMatrix),
                glm::make_vec4(lighPosInEyeSpace),
                glm::make_vec3(cameraPos));

    env->ReleaseFloatArrayElements(mEyeProjectionMatrix_, mEyeProjectionMatrix, 0);
    env->ReleaseFloatArrayElements(mEyeViewMatrix_, mEyeViewMatrix, 0);
    env->ReleaseFloatArrayElements(myLighPosInEyeSpace_, myLighPosInEyeSpace, 0);
    env->ReleaseFloatArrayElements(mCameraPos_, mCameraPos, 0);
    delete eyeProjectionMatrix;
    delete eyeViewMatrix;
    delete lighPosInEyeSpace;
    delete cameraPos;
}