//
// Created by samuel on 30/05/18.
//

#include "wrapper_utils.h"

float* jfloatPtrToCppFloatPtr(jfloat* array, int length) {
    float* res = new float[length];
    for (int i = 0; i < length; i++) {
        res[i] = array[i];
    }
    return res;
}