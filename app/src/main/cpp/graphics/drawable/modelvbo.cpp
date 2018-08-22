#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <glm/gtc/type_ptr.hpp>

#include "modelvbo.h"
#include "../../utils/string_utils.h"


ModelVBO::ModelVBO(string obj_file_text) {
	init();
	bind();
	bindBuffer(parseObj(obj_file_text));

	lightCoef = 1;
	distanceCoef = 0;

	for (int i = 0; i < 3; i++)
		this->color[i] = rand() / RAND_MAX;
	this->color[3] = 1.f;
}

ModelVBO::ModelVBO(string obj_file_text, float color[4]) {
	init();
	bind();
	bindBuffer(parseObj(obj_file_text));

	lightCoef = 1;
	distanceCoef = 0;

	for (int i = 0; i < 4; i++) {
		this->color[i] = color[i];
	}
}

std::vector<float> ModelVBO::parseObj(string obj_file_text) {
	nbVertex = 0;

	vector<std::string> lines = split(obj_file_text, '\n');

	vector<float> vertex_list;
	vector<float> normal_list;
	vector<int> vertex_draw_order;
	vector<int> normal_draw_order;

	for (auto str : lines) {
		vector<std::string> splitted_line = split(str, ' ');
		if (!splitted_line.empty()) {
			if (splitted_line[0] == "vn") {
				normal_list.push_back(std::stof(splitted_line[1]));
				normal_list.push_back(std::stof(splitted_line[2]));
				normal_list.push_back(std::stof(splitted_line[3]));
			} else if (splitted_line[0] == "v") {
				vertex_list.push_back(std::stof(splitted_line[1]));
				vertex_list.push_back(std::stof(splitted_line[2]));
				vertex_list.push_back(std::stof(splitted_line[3]));
			} else if (splitted_line[0] == "f") {
				vector<string> v1 = split(splitted_line[1], '/');
				vector<string> v2 = split(splitted_line[2], '/');
				vector<string> v3 = split(splitted_line[3], '/');

				vertex_draw_order.push_back(std::stoi(v1[0]));
				vertex_draw_order.push_back(std::stoi(v2[0]));
				vertex_draw_order.push_back(std::stoi(v3[0]));

				normal_draw_order.push_back(std::stoi(v1[2]));
				normal_draw_order.push_back(std::stoi(v2[2]));
				normal_draw_order.push_back(std::stoi(v3[2]));

				v1.clear();
				v2.clear();
				v3.clear();
			}
		}
		splitted_line.clear();
	}

	vector<float> packedData;

	for (int i = 0; i < vertex_draw_order.size(); i++) {
		packedData.push_back(vertex_list[(vertex_draw_order[i] - 1) * 3]);
		packedData.push_back(vertex_list[(vertex_draw_order[i] - 1) * 3 + 1]);
		packedData.push_back(vertex_list[(vertex_draw_order[i] - 1) * 3 + 2]);

		packedData.push_back(normal_list[(normal_draw_order[i] - 1) * 3]);
		packedData.push_back(normal_list[(normal_draw_order[i] - 1) * 3 + 1]);
		packedData.push_back(normal_list[(normal_draw_order[i] - 1) * 3 + 2]);
		nbVertex++;
	}

	vertex_list.clear();
	vertex_draw_order.clear();
	normal_list.clear();
	normal_draw_order.clear();

	return packedData;
}

TransparentModelVBO::TransparentModelVBO(const string &obj_file_text, float *color) : ModelVBO(obj_file_text, color) {}

TransparentModelVBO::TransparentModelVBO(const string &obj_file_text) : ModelVBO(obj_file_text) {}

void TransparentModelVBO::draw(glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos) {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	DiffuseModel::draw(mvp_matrix, mv_matrix, light_pos);
	glDisable(GL_BLEND);
}
