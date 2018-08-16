//
// Created by samuel on 30/05/18.
//

#include <jni.h>
#include "../entity/ground/map.h"
#include "../core/engine.h"

extern "C"
JNIEXPORT jlong JNICALL
Java_com_samuelberrien_phyvr_wrappers_MainWrappers_initEngine(JNIEnv *env, jobject instance, jlong boxesPtr) {

	vector<Base *> *boxes = (vector<Base *> *) boxesPtr;

	glm::vec3 start(-1000.f, -200.f, -1000.f);
	glm::vec3 end(1000.f, 200.f, 1000.f);

	// TODO modifier wrapper pr limit level ? (dyn_cast deg...)
	/*for (Base *b : *boxes)
		if(Map* m = dynamic_cast<Map*>(b)) {
			start = m->getMinPos();
			end = m->getMaxPos();
		}
	end.y = 200.f;*/
	Engine *level = new Engine(boxes, new BoxLimits(start, end - start));

	return (long) level;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_wrappers_MainWrappers_updateEngine(JNIEnv *env, jobject instance, jlong levelptr) {
	Engine *level = (Engine *) levelptr;
	level->update(1.f / 60.f);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_samuelberrien_phyvr_wrappers_MainWrappers_freeLevel(JNIEnv *env, jobject instance, jlong levelPtr) {
	delete (Engine *) levelPtr;
}