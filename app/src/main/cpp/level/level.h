//
// Created by samuel on 25/05/18.
//

#ifndef PHYVR_LEVEL_H
#define PHYVR_LEVEL_H


class level {
public:
    void update(float mHeadView[16]);
    void draw(float mEyeProjectionMatrix[16], float mEyeViewMatrix[16], float myLighPosInEyeSpace[4], float mCameraPos[3]);
private:
    void updateLight(float xyz[3]);
};


#endif //PHYVR_LEVEL_H
