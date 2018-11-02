//
// Created by samuel on 29/10/18.
//

#ifndef PHYVR_NORMALMODEL_H
#define PHYVR_NORMALMODEL_H

#include <string>
#include <GLES3/gl3.h>
#include <android/asset_manager.h>
#include <glm/glm.hpp>
#include "../misc.h"

using namespace std;

class NormalMapModel : public GLDrawable {
private:
	const int POSITION_SIZE = 3;
	const int NORMAL_SIZE = 3;
	const int TEX_COORD_SIZE = 2;
	const int BYTES_PER_FLOAT = 4;
	const int STRIDE = (POSITION_SIZE + NORMAL_SIZE + TEX_COORD_SIZE) * BYTES_PER_FLOAT;

	GLuint mProgram;
	GLuint mPositionHandle;
	GLuint mNormalHandle;
	GLuint mTextCoordHandle;
	GLuint mMVPMatrixHandle;
	GLuint mLightPosHandle;
	GLuint mMVMatrixHandle;
	GLuint mTexHandle;
	GLuint mNormalMapHandle;

	GLuint buffer;

	GLuint *textures;

	int nbVertex;

	void initPrgm(AAssetManager *mgr);

	void initTex(AAssetManager *mgr, string textureFile, string normalsFile);

	void bind();

	void genBuffer(AAssetManager *mgr, string objFile);

	vector<float> parseObj(AAssetManager *mgr, string objFile);

public:
	NormalMapModel(AAssetManager *mgr, string objFile, string textureFile, string normalsFile);

	void draw(gl_draw_info info) override ;

	virtual ~NormalMapModel();
};


#endif //PHYVR_NORMALMODEL_H
