//
// Created by samuel on 23/08/18.
//

#include "cubemap.h"
#include "../../utils/shader.h"
#include "../../utils/assets.h"
#include "../../utils/gl_utils.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

string cube_vs = "attribute vec3 a_vp;\n"
		"uniform mat4 u_MVPMatrix;\n"
		"varying vec3 v_tex_coords;\n"
		"void main () {\n"
		"  v_tex_coords = a_vp;\n"
		"  gl_Position = u_MVPMatrix * vec4(a_vp, 1.0);\n"
		"}";

string cube_fs = "precision mediump float;\n"
		"varying vec3 v_tex_coords;\n"
		"uniform samplerCube u_cube_map;\n"
		"void main () {\n"
		"  gl_FragColor = textureCube(u_cube_map, v_tex_coords);\n"
		"}";

CubeMap::CubeMap(AAssetManager *mgr, string cubeMapAssetPath, float sideLength)
		: modelMatrix(glm::scale(glm::mat4(1.f), glm::vec3(sideLength))) {
	makeProgram();
	bind();
	loadCubeMaptexture(mgr, cubeMapAssetPath);
}

void CubeMap::draw(draw_infos infos) {
	glm::mat4 mvpMatrix = infos.proj_matrix * infos.view_matrix * modelMatrix;

	glUseProgram(mProgram);

	glUniformMatrix4fv(mvpMatrixHandle, 1, GL_FALSE, glm::value_ptr(mvpMatrix));

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, textures);
	glUniform1i(samplerCubeHandle, 0);

	glVertexAttribPointer(texCoordHandle, 3, GL_FLOAT, GL_FALSE, 0, vertices);
	glEnableVertexAttribArray(texCoordHandle);

	glDrawArrays(GL_TRIANGLES, 0, nbTriangle);

	glDisableVertexAttribArray(texCoordHandle);

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void CubeMap::bind() {
	texCoordHandle = (GLuint) glGetAttribLocation(mProgram, "a_vp");
	mvpMatrixHandle = (GLuint) glGetUniformLocation(mProgram, "u_MVPMatrix");
	samplerCubeHandle = (GLuint) glGetUniformLocation(mProgram, "u_cube_map");
}

void CubeMap::makeProgram() {
	vertexShader = loadShader(GL_VERTEX_SHADER, cube_vs.c_str());
	fragmentShader = loadShader(GL_FRAGMENT_SHADER, cube_fs.c_str());

	mProgram = glCreateProgram();             // create empty OpenGL Program
	glAttachShader(mProgram, vertexShader);   // add the vertex shader to program
	glAttachShader(mProgram, fragmentShader); // add the fragment shader to program
	glLinkProgram(mProgram);
}

void CubeMap::loadCubeMaptexture(AAssetManager *mgr, string cubaMapAssetPath) {
	glGenTextures(1, &textures);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, textures);

	GLenum idTxt[]{
			GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
			GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
	};
	string fileName[]{
			"posx.png", "posy.png", "posz.png", "negx.png", "negy.png", "negz.png"
	};
	for (int i = 0; i < 6; i++) {
		libpng_image tmp = readPNG(mgr, cubaMapAssetPath + fileName[i]);
		imgRGB img = toImgRGB(tmp);
		glTexImage2D(idTxt[i], 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img.pixels);
		delete[] tmp.data;
		delete[] tmp.rowPtrs;
		delete[] img.pixels;
	}

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

CubeMap::~CubeMap() {
	glDeleteTextures(1, &textures);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	glDeleteProgram(mProgram);
}
