//
// Created by samuel on 28/05/18.
//

#ifndef PHYVR_DIFFUSEMODEL_H
#define PHYVR_DIFFUSEMODEL_H


#include <GLES3/gl3.h>
#include <vector>
#include <glm/glm.hpp>
#include "../misc.h"

class DiffuseModel : public GLDrawable {
private:
	// Data sizes
	const int POSITION_SIZE = 3;
	const int NORMAL_SIZE = 3;
	const int BYTES_PER_FLOAT = 4;
	const int STRIDE = (POSITION_SIZE + NORMAL_SIZE) * BYTES_PER_FLOAT;

	// GPU program handle
	GLuint mProgram;
	GLuint vertexShader;
	GLuint fragmentShader;

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
	void draw(gl_draw_info info) override ;

	virtual ~DiffuseModel();
};

#endif //PHYVR_DIFFUSEMODEL_H
