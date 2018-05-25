//
// Created by samuel on 19/01/18.
//
#ifndef PHYVR_OBJVBO_H
#define PHYVR_OBJVBO_H

#include <GLES2/gl2.h>

using namespace std;

/**
 * One color object with VBO
 * Use OBJ or STL file
 */
class ModelVBO {

private:
    // Data sizes
    const int POSITION_SIZE = 3;
    const int NORMAL_SIZE = 3;
    const int BYTES_PER_FLOAT = 4;
    const int STRIDE = (POSITION_SIZE + NORMAL_SIZE) * BYTES_PER_FLOAT;

    // GPU program handle
    GLuint mProgram;
    GLuint mPositionHandle;
    GLuint mNormalHandle;
    GLuint mColorHandle;
    GLuint mMVPMatrixHandle;
    GLuint mLightPosHandle;
    GLuint mMVMatrixHandle;
    GLuint mDistanceCoefHandle;
    GLuint mLightCoefHandle;

    // Number of vertex -> needed for draw
    int nbVertex;

    float color[4];

    float distanceCoef;
    float lightCoef;

    GLuint packedDataBufferId;

    void init();
    void bind();
    void bindBuffer(std::vector<float> packedData);
    std::vector<float> parseObj(std::string obj_file_text);

public:
    ModelVBO(std::string obj_file_text);
    ModelVBO(std::string obj_file_text, float color[4]);
    void draw(float mvp_matrix[16], float mv_matrix[16], float light_pos[3]);
};

#endif
