//
// Created by samuel on 19/01/18.
//
#ifndef PHYVR_OBJVBO_H
#define PHYVR_OBJVBO_H

#include <GLES2/gl2.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>

#include "diffusemodel.h"

using namespace std;

/**
 * One color object with VBO
 * Use OBJ or STL file
 */
class ModelVBO : public DiffuseModel {
	std::vector<float> parseObj(std::string obj_file_text);

public:
	ModelVBO(std::string obj_file_text);

	ModelVBO(std::string obj_file_text, float color[4]);
};

class TransparentModelVBO : public ModelVBO {
public:
	TransparentModelVBO(const string &obj_file_text, float *color);

	TransparentModelVBO(const string &obj_file_text);

	void draw(glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos) override ;
};

#endif
