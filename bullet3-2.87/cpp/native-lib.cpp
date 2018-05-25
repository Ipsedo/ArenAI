#include <jni.h>
#include <string>
#include <btBulletDynamicsCommon.h>
#include <GLES2/gl2.h>

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
