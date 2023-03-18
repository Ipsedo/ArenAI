//
// Created by samuel on 19/01/18.
//
#ifndef PHYVR_OBJVBO_H
#define PHYVR_OBJVBO_H

#include <GLES3/gl3.h>
#include "glm/glm.hpp"
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

	ModelVBO(std::string obj_file_text, float r, float g, float b, float a);
};

class TransparentModelVBO : public ModelVBO {
public:
	TransparentModelVBO(const string &obj_file_text, float r, float g, float b, float a);

	TransparentModelVBO(const string &obj_file_text);

	void draw(gl_draw_info info) override;
};

#endif
