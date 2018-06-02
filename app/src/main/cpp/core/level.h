//
// Created by samuel on 02/06/18.
//

#ifndef PHYVR_LEVEL_H
#define PHYVR_LEVEL_H


#include "../entity/player.h"

class Level {
public:
	Level();
	virtual void init();

	Player* getPlayer();
	vector<Base*>* getEntities();

	virtual bool won() = 0;
	virtual bool lose() = 0;

private:
	bool isInit;
	vector<Base*> entities;
	Player* player;
};


#endif //PHYVR_LEVEL_H
