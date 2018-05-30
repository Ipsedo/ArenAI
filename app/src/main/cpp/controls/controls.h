//
// Created by samuel on 28/05/18.
//

#ifndef PHYVR_CONTROLS_H
#define PHYVR_CONTROLS_H


#include <tuple>

class Controls {
private:
	float joystick1Ver;
	float joystick1Hor;

public:
	std::tuple<int, int> getJoystick1();

	std::tuple<int, int> getJoystick2();

	bool isPressingFire();

	bool isPressingBrake();
};


#endif //PHYVR_CONTROLS_H
