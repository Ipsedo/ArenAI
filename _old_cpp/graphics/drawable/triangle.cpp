//
// Created by samuel on 06/09/18.
//

#include "triangle.h"
#include <string>
#include "glm/gtc/type_ptr.hpp"
#include "../../utils/shader.h"

using namespace std;

string simple_vs = "uniform mat4 uMVPMatrix;\n"
				   "attribute vec4 vPosition;\n"
				   "\n"
				   "void main() {\n"
				   "    gl_Position = uMVPMatrix * vPosition;\n"
				   "}";

string simple_fs = "precision mediump float;\n"
				   "\n"
				   "uniform vec4 vColor;\n"
				   "\n"
				   "void main() {\n"
				   "    gl_FragColor = vColor;\n"
				   "}";

void Triangle::makeProgram() {
	mProgram = glCreateProgram();             // create empty OpenGL Program
	glAttachShader(mProgram, vertexShader);   // add the vertex shader to program
	glAttachShader(mProgram, fragmentShader); // add the fragment shader to program
	glLinkProgram(mProgram);
}

void Triangle::bind() {
	mPositionHandle = (GLuint) glGetAttribLocation(mProgram, "vPosition");
	mColorHandle = (GLuint) glGetUniformLocation(mProgram, "vColor");
	mMVPMatrixHandle = (GLuint) glGetUniformLocation(mProgram, "uMVPMatrix");
}

Triangle::Triangle(float r, float g, float b)
		: vertexShader(loadShader(GL_VERTEX_SHADER, simple_vs.c_str())),
		  fragmentShader(loadShader(GL_FRAGMENT_SHADER, simple_fs.c_str())),
		  color{r, g, b, 1.f} {
	makeProgram();
	bind();
}

void Triangle::draw(glm::mat4 MVPMatrix) {
	glUseProgram(mProgram);
	glEnableVertexAttribArray(mPositionHandle);
	glVertexAttribPointer(mPositionHandle, 3, GL_FLOAT, GL_FALSE, 3 * 4, vertex);
	glUniform4fv(mColorHandle, 1, color);
	glUniformMatrix4fv(mMVPMatrixHandle, 1, GL_FALSE, glm::value_ptr(MVPMatrix));
	glDrawArrays(GL_TRIANGLES, 0, 9 / 3);
}

Triangle::~Triangle() {
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	glDeleteProgram(mProgram);
}
