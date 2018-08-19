//
// Created by samuel on 30/05/18.
//

#include <jni.h>
#include "wrapper_utils.h"
#include <glm/gtc/type_ptr.hpp>
#include "../graphics/renderer.h"

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_wrappers_MainWrappers_makeRenderer(JNIEnv *env, jobject instance, jlong levelPtr) {

	Level *level = (Level *) levelPtr;
	Renderer *renderer = new Renderer(level);

	return (long) renderer;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_wrappers_MainWrappers_willDrawRenderer(JNIEnv *env, jobject instance,
																	jlong rendererPtr, jfloatArray mHeadView_,
																	jboolean VR) {
	jfloat *mHeadView = env->GetFloatArrayElements(mHeadView_, NULL);
	float *headView = jfloatPtrToCppFloatPtr(mHeadView, 16);

	Renderer *renderer = (Renderer *) rendererPtr;
	renderer->update(glm::make_mat4(headView), VR);

	env->ReleaseFloatArrayElements(mHeadView_, mHeadView, 0);
	delete[] headView;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_wrappers_MainWrappers_drawRenderer(JNIEnv *env, jobject instance,
																jlong rendererPtr,
																jfloatArray mEyeProjectionMatrix_,
																jfloatArray mEyeViewMatrix_,
																jfloatArray myLighPosInEyeSpace_,
																jfloatArray mCameraPos_) {
	jfloat *mEyeProjectionMatrix = env->GetFloatArrayElements(mEyeProjectionMatrix_, NULL);
	jfloat *mEyeViewMatrix = env->GetFloatArrayElements(mEyeViewMatrix_, NULL);
	jfloat *myLighPosInEyeSpace = env->GetFloatArrayElements(myLighPosInEyeSpace_, NULL);
	jfloat *mCameraPos = env->GetFloatArrayElements(mCameraPos_, NULL);

	float *eyeProjectionMatrix = jfloatPtrToCppFloatPtr(mEyeProjectionMatrix, 16);
	float *eyeViewMatrix = jfloatPtrToCppFloatPtr(mEyeViewMatrix, 16);
	float *lighPosInEyeSpace = jfloatPtrToCppFloatPtr(myLighPosInEyeSpace, 4);
	float *cameraPos = jfloatPtrToCppFloatPtr(mCameraPos, 3);

	Renderer *renderer = (Renderer *) rendererPtr;
	renderer->draw(glm::make_mat4(eyeProjectionMatrix),
				   glm::make_mat4(eyeViewMatrix),
				   glm::make_vec4(lighPosInEyeSpace),
				   glm::make_vec3(cameraPos));

	env->ReleaseFloatArrayElements(mEyeProjectionMatrix_, mEyeProjectionMatrix, 0);
	env->ReleaseFloatArrayElements(mEyeViewMatrix_, mEyeViewMatrix, 0);
	env->ReleaseFloatArrayElements(myLighPosInEyeSpace_, myLighPosInEyeSpace, 0);
	env->ReleaseFloatArrayElements(mCameraPos_, mCameraPos, 0);
	delete[] eyeProjectionMatrix;
	delete[] eyeViewMatrix;
	delete[] lighPosInEyeSpace;
	delete[] cameraPos;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_wrappers_MainWrappers_freeRenderer(JNIEnv *env, jobject instance, jlong rendererPtr) {
	delete (Renderer *) rendererPtr;
}