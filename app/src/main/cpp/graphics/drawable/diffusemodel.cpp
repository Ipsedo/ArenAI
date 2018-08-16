//
// Created by samuel on 28/05/18.
//

#include <string>
#include <glm/gtc/type_ptr.hpp>
#include "diffusemodel.h"
#include "../../utils/shader.h"


std::string vs = "uniform mat4 u_MVPMatrix;\n"
		"uniform mat4 u_MVMatrix;\n"
		"uniform vec4 v_Color;\n"
		"attribute vec4 a_Position;\n"
		"attribute vec3 a_Normal;\n"
		"varying vec3 v_Position;\n"
		"varying vec3 v_Normal;\n"
		"void main(){\n"
		"    v_Position = vec3(u_MVMatrix * a_Position);\n"
		"    v_Normal = normalize(vec3(u_MVMatrix * vec4(a_Normal, 0.0)));\n"
		"    gl_Position = u_MVPMatrix * a_Position;\n"
		"}";

std::string fs = "precision mediump float;\n"
		"uniform vec3 u_LightPos;\n"
		"uniform float u_distance_coef;\n"
		"uniform float u_light_coef;\n"
		"uniform vec4 u_Color;\n"
		"varying vec3 v_Position;\n"
		"varying vec3 v_Normal;\n"
		"void main(){\n"
		"    float distance = length(u_LightPos - v_Position);\n"
		"    vec3 lightVector = normalize(u_LightPos - v_Position);\n"
		"    float diffuse = max(dot(v_Normal, lightVector), 0.1) * u_light_coef;\n"
		"    diffuse = diffuse * (1.0 / (1.0 + (u_distance_coef * distance * distance)));\n"
		"    gl_FragColor = u_Color * diffuse;\n"
		"}";

void DiffuseModel::init() {
	mProgram = glCreateProgram();
	vertexShader = loadShader(GL_VERTEX_SHADER, vs.c_str());
	fragmentShader = loadShader(GL_FRAGMENT_SHADER, fs.c_str());
	glAttachShader(mProgram, vertexShader);
	glAttachShader(mProgram, fragmentShader);
	glLinkProgram(mProgram);
}

void DiffuseModel::bind() {
	mMVPMatrixHandle = (GLuint) glGetUniformLocation(mProgram, "u_MVPMatrix");
	mMVMatrixHandle = (GLuint) glGetUniformLocation(mProgram, "u_MVMatrix");
	mPositionHandle = (GLuint) glGetAttribLocation(mProgram, "a_Position");
	mColorHandle = (GLuint) glGetUniformLocation(mProgram, "u_Color");
	mLightPosHandle = (GLuint) glGetUniformLocation(mProgram, "u_LightPos");
	mDistanceCoefHandle = (GLuint) glGetUniformLocation(mProgram, "u_distance_coef");
	mLightCoefHandle = (GLuint) glGetUniformLocation(mProgram, "u_light_coef");
	mNormalHandle = (GLuint) glGetAttribLocation(mProgram, "a_Normal");
}

void DiffuseModel::bindBuffer(std::vector<float> packedData) {
	glGenBuffers(1, &packedDataBufferId);

	glBindBuffer(GL_ARRAY_BUFFER, packedDataBufferId);
	glBufferData(GL_ARRAY_BUFFER, packedData.size() * BYTES_PER_FLOAT, &packedData[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	packedData.clear();
}

void DiffuseModel::draw(glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos) {
	glUseProgram(mProgram);

	glBindBuffer(GL_ARRAY_BUFFER, packedDataBufferId);
	glEnableVertexAttribArray(mPositionHandle);
	glVertexAttribPointer(mPositionHandle, POSITION_SIZE, GL_FLOAT, GL_FALSE,
						  STRIDE, 0);

	glEnableVertexAttribArray(mNormalHandle);
	glVertexAttribPointer(mNormalHandle, NORMAL_SIZE, GL_FLOAT, GL_FALSE,
						  STRIDE, (char *) NULL + POSITION_SIZE * BYTES_PER_FLOAT);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glUniformMatrix4fv(mMVPMatrixHandle, 1, GL_FALSE, glm::value_ptr(mvp_matrix));

	glUniformMatrix4fv(mMVMatrixHandle, 1, GL_FALSE, glm::value_ptr(mv_matrix));

	glUniform3fv(mLightPosHandle, 1, glm::value_ptr(light_pos));

	glUniform4fv(mColorHandle, 1, color);

	glUniform1f(mDistanceCoefHandle, distanceCoef);

	glUniform1f(mLightCoefHandle, lightCoef);

	glDrawArrays(GL_TRIANGLES, 0, nbVertex);

	glDisableVertexAttribArray(mPositionHandle);
	glDisableVertexAttribArray(mNormalHandle);
}

DiffuseModel::~DiffuseModel() {
	glDeleteBuffers(1, &packedDataBufferId);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	glDeleteProgram(mProgram);
}