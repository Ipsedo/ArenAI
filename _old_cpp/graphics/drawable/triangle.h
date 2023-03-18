//
// Created by samuel on 06/09/18.
//

#ifndef PHYVR_TRIANGLE_H
#define PHYVR_TRIANGLE_H

#include <math.h>
#include "glm/glm.hpp"
#include <GLES3/gl3.h>

class Triangle {
private:
	const float vertex[9]
			{0.f, 1.f, 0.f,
			 (float) cos(-M_PI / 3.), (float) sin(-M_PI / 3.), 0.f,
			 (float) cos(-M_PI * 2. / 3.), (float) sin(-M_PI * 2. / 3.), 0.f};

	GLuint mPositionHandle;
	GLuint mColorHandle;
	GLuint mMVPMatrixHandle;
	GLuint mProgram;

	GLuint vertexShader;
	GLuint fragmentShader;

	float color[4];

	void makeProgram();

	void bind();

public:
	Triangle(float r, float g, float b);

	void draw(glm::mat4 MVPMatrix);

	~Triangle();

};


#endif //PHYVR_TRIANGLE_H
