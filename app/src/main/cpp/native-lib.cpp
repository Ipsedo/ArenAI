#include <jni.h>
#include <string>
#include <btBulletDynamicsCommon.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include "utils/assets.h"
#include "graphics/renderer.h"
#include "level/level.h"
#include "entity/convex.h"
#include "entity/ground/map.h"
#include <glm/gtc/type_ptr.hpp>

#define HEIGHT_SPAWN 30.f

extern "C"
JNIEXPORT jstring JNICALL
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
 * Renderer stuff
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
Java_com_samuelberrien_phyvr_MyGvrView_initBoxes(JNIEnv *env, jobject instance,
                                                 jobject assetManager) {

    AAssetManager* cppMgr = AAssetManager_fromJava(env, assetManager);

    glm::mat4 id(1.f);

    Base* box = new Box(cppMgr, glm::vec3(0.f, HEIGHT_SPAWN, 5.f), glm::vec3(1.f,1.f,1.f), id, 1.f);
    Base* sol = new Box(cppMgr, glm::vec3(0.f, -5.f, 0.f), glm::vec3(4.5f,0.1f,4.5f), id, 0.f);

    vector<Base*>* boxes = new vector<Base*>();
    boxes->push_back(box);
    boxes->push_back(sol);

    return (long) boxes;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_initEntity(JNIEnv *env, jobject instance,
                                                  jobject assetManager, jfloatArray heightmap_,
                                                  jint width, jint height) {
    jfloat *heightmap = env->GetFloatArrayElements(heightmap_, NULL);
    float *map = jfloatPtrToCppFloatPtr(heightmap, width * height);
    float max = -std::numeric_limits<float>::max(),
        min = std::numeric_limits<float>::max();
    for (int i = 0; i < width * height; i++) {
        min = min > map[i] ? map[i] : min;
        max = max < map[i] ? map[i] : max;
    }
    __android_log_print(ANDROID_LOG_DEBUG, "SAM", "max : %f, min : %f", max, min);

    AAssetManager* cppMgr = AAssetManager_fromJava(env, assetManager);

    glm::mat4 id(1.f);

    Base* box = new Box(cppMgr, glm::vec3(0.f, HEIGHT_SPAWN, 5.f), glm::vec3(1.f,1.f,1.f), id, 1.f);
    Base* sol = new Map(glm::vec3(0.f, -5.f, 0.f), width, height, max, map, glm::vec3(100.f,HEIGHT_SPAWN,100.f));
            //new Box(cppMgr, glm::vec3(0.f, -5.f, 0.f), glm::vec3(4.5f,0.1f,4.5f), id, 0.f);

    vector<Base*>* boxes = new vector<Base*>();
    boxes->push_back(box);
    boxes->push_back(sol);

    return (long) boxes;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_initLevel(JNIEnv *env, jobject instance, jlong boxesPtr) {

    vector<Base*>* boxes = (vector<Base*>*) boxesPtr;

    Level* level = new Level(boxes);

    return (long)level;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_initRenderer(JNIEnv *env, jobject instance, jlong boxesPtr) {

    vector<Base*>* boxes = (vector<Base*>*) boxesPtr;
    Renderer* renderer = new Renderer(boxes);

    return (long)renderer;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_willDrawRenderer(JNIEnv *env, jobject instance,
                                                        jlong rendererPtr, jfloatArray mHeadView_) {
    jfloat *mHeadView = env->GetFloatArrayElements(mHeadView_, NULL);
    float *headView = jfloatPtrToCppFloatPtr(mHeadView, 16);

    Renderer* renderer = (Renderer*) rendererPtr;
    renderer->update(glm::make_mat4(headView));

    env->ReleaseFloatArrayElements(mHeadView_, mHeadView, 0);
    delete headView;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_updateLevel(JNIEnv *env, jobject instance, jlong levelptr) {
    Level* level = (Level*) levelptr;
    level->update(1.f / 60.f);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_drawRenderer(JNIEnv *env, jobject instance,
                                                    jlong rendererPtr,
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

    Renderer* renderer = (Renderer*) rendererPtr;
    renderer->draw(glm::make_mat4(eyeProjectionMatrix),
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

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_addBox(JNIEnv *env, jobject instance, jobject assetManager,
                                              jlong boxesPtr, jlong levelPtr) {

    vector<Base*>* boxes = (vector<Base*>*) boxesPtr;
    Level* level = (Level*) levelPtr;

    AAssetManager* cppMgr = AAssetManager_fromJava(env, assetManager);

    glm::mat4 id(1.f);
    double r = (double) rand() / RAND_MAX;

    if (r < 1. / 120.) {
        float x = 5.f * (float) rand() / RAND_MAX;
        float z = 5.f * (float) rand() / RAND_MAX;
        float scale = 2.f * (float) rand() / RAND_MAX;
        Base *base;
        if ((float) rand() / RAND_MAX > 0.8) {
            base = new Box(cppMgr,
                          glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale),
                          id, 1.f);
        } else {
            /*std::string m =
                    (float) rand() / RAND_MAX > 0.5 ? "obj/icosahedron.obj" : "obj/ast1.obj";*/
            std::string m = "obj/icosahedron.obj";
            base = new Convex(cppMgr, m,
                             glm::vec3(x, HEIGHT_SPAWN, z), glm::vec3(scale),
                             id, 1.f);
        }
        boxes->push_back(base);
        level->addNewBox(base);
    }

}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_freeBoxes(JNIEnv *env, jobject instance, jlong boxesPtr) {

    vector<Base*>* boxes = (vector<Base*>*) boxesPtr;
    for (Base* b : *boxes)
        delete b;
    boxes->clear();
    delete boxes;

}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_freeLevel(JNIEnv *env, jobject instance, jlong levelPtr) {
    delete (Level*) levelPtr;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_MyGvrView_freeRenderer(JNIEnv *env, jobject instance, jlong rendererPtr) {
    delete (Renderer*) rendererPtr;
}