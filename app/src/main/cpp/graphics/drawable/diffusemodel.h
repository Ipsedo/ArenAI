//
// Created by samuel on 28/05/18.
//

#ifndef PHYVR_DIFFUSEMODEL_H
#define PHYVR_DIFFUSEMODEL_H


#include <GLES2/gl2.h>
#include <vector>
#include <glm/glm.hpp>

class DiffuseModel {
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

	GLuint packedDataBufferId;

protected:
	float distanceCoef;
	float lightCoef;

	float color[4];

	// Number of vertex -> needed for draw
	int nbVertex;

	void init();

	void bind();

	void bindBuffer(std::vector<float> packedData);

public:
	void draw(glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos);
};


#endif //PHYVR_DIFFUSEMODEL_H
