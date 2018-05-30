//
// Created by samuel on 28/05/18.
//

#ifndef PHYVR_CONTROLS_H
#define PHYVR_CONTROLS_H


#include <tuple>

class Controls {
public:
	virtual void onInput(float xAxis, float speed, bool brake) = 0;
};


#endif //PHYVR_CONTROLS_H
