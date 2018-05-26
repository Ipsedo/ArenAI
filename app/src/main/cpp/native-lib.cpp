#include <jni.h>
#include <string>
#include <btBulletDynamicsCommon.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "utils/assets.h"

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