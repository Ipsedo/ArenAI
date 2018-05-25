#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <glm/gtc/type_ptr.hpp>

#include "modelvbo.h"
#include "../utils/shader.h"
#include "../utils/string_utils.h"

ModelVBO::ModelVBO(string obj_file_text) {
    init();
    bind();
    bindBuffer(parseObj(obj_file_text));

    lightCoef = 1;
    distanceCoef = 0;
    this->color[0] = rand() / RAND_MAX;
    this->color[1] = rand() / RAND_MAX;
    this->color[2] = rand() / RAND_MAX;
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

void ModelVBO::init() {
    mProgram = glCreateProgram();
    // TODO add shader code
    GLuint vertexShader = loadShader(GL_VERTEX_SHADER, "/shaders/diffuse_vs.glsl");
    GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, "/shaders/diffuse_fs.glsl");
    glAttachShader(mProgram, vertexShader);
    glAttachShader(mProgram, fragmentShader);
    glLinkProgram(mProgram);
}

void ModelVBO::bind() {
    mMVPMatrixHandle = (GLuint) glGetUniformLocation(mProgram, "u_MVPMatrix");
    mMVMatrixHandle = (GLuint) glGetUniformLocation(mProgram, "u_MVMatrix");
    mPositionHandle = (GLuint) glGetAttribLocation(mProgram, "a_Position");
    mColorHandle = (GLuint) glGetUniformLocation(mProgram, "u_Color");
    mLightPosHandle = (GLuint) glGetUniformLocation(mProgram, "u_LightPos");
    mDistanceCoefHandle = (GLuint) glGetUniformLocation(mProgram, "u_distance_coef");
    mLightCoefHandle = (GLuint) glGetUniformLocation(mProgram, "u_light_coef");
    mNormalHandle = (GLuint) glGetAttribLocation(mProgram, "a_Normal");
}

void ModelVBO::bindBuffer(std::vector<float> packedData) {
    glGenBuffers(1, &packedDataBufferId);

    glBindBuffer(GL_ARRAY_BUFFER, packedDataBufferId);
    glBufferData(GL_ARRAY_BUFFER, packedData.size() * BYTES_PER_FLOAT, &packedData[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    packedData.clear();
}

std::vector<float> ModelVBO::parseObj(string obj_file_text) {
    nbVertex = 0;

    vector<std::string> lines = split(obj_file_text, '\n');

    vector<float> vertex_list;
    vector<float> normal_list;
    vector<int> vertex_draw_order;
    vector<int> normal_draw_order;

    for (auto str : lines) {
        //std::cout << str << std::endl;
        vector<std::string> splitted_line = split(str, ' ');
        if(!splitted_line.empty()) {
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

void ModelVBO::draw(glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos) {
    glUseProgram(mProgram);

    glBindBuffer(GL_ARRAY_BUFFER, packedDataBufferId);
    glEnableVertexAttribArray(mPositionHandle);
    glVertexAttribPointer(mPositionHandle, POSITION_SIZE, GL_FLOAT, GL_FALSE,
            STRIDE, 0);

    glEnableVertexAttribArray(mNormalHandle);
    glVertexAttribPointer(mNormalHandle, NORMAL_SIZE, GL_FLOAT, GL_FALSE,
            STRIDE, (char *)NULL + POSITION_SIZE * BYTES_PER_FLOAT);

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
