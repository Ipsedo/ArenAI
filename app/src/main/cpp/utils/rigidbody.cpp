//
// Created by samuel on 29/05/18.
//

#include "rigidbody.h"
#include "string_utils.h"

#include <btBulletDynamicsCommon.h>

tuple<btRigidBody::btRigidBodyConstructionInfo, btDefaultMotionState *>
localCreateInfo(btScalar mass, const btTransform &startTransform, btCollisionShape *shape) {
	btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

	//rigidbody is dynamic if and only if mass is non zero, otherwise static
	bool isDynamic = (mass != 0.f);

	btVector3 localInertia(0, 0, 0);
	if (isDynamic)
		shape->calculateLocalInertia(mass, localInertia);

	btDefaultMotionState *myMotionState = new btDefaultMotionState(startTransform);

	return std::tuple<btRigidBody::btRigidBodyConstructionInfo, btDefaultMotionState *>
			(btRigidBody::btRigidBodyConstructionInfo(mass, myMotionState, shape, localInertia), myMotionState);
}

btConvexHullShape *parseObj(std::string objFileText) {
	vector<std::string> lines = split(objFileText, '\n');
	btConvexHullShape *shape = new btConvexHullShape();
	vector<float> vertex_list;
	vector<int> vertex_draw_order;

	for (auto str : lines) {
		vector<std::string> splitted_line = split(str, ' ');
		if (!splitted_line.empty()) {
			if (splitted_line[0] == "v") {
				vertex_list.push_back(std::stof(splitted_line[1]));
				vertex_list.push_back(std::stof(splitted_line[2]));
				vertex_list.push_back(std::stof(splitted_line[3]));
			} else if (splitted_line[0] == "f") {
				vector<string> v1 = split(splitted_line[1], '/');
				vector<string> v2 = split(splitted_line[2], '/');
				vector<string> v3 = split(splitted_line[3], '/');

				vertex_draw_order.push_back(std::stoi(v1[0]) - 1);
				vertex_draw_order.push_back(std::stoi(v2[0]) - 1);
				vertex_draw_order.push_back(std::stoi(v3[0]) - 1);

				v1.clear();
				v2.clear();
				v3.clear();
			}
		}
		splitted_line.clear();
	}

	unsigned long nbVertex = vertex_draw_order.size();
	for (int i = 0; i < nbVertex; i++) {
		btVector3 point(vertex_list[(vertex_draw_order[i] - 1) * 3],
						vertex_list[(vertex_draw_order[i] - 1) * 3 + 1],
						vertex_list[(vertex_draw_order[i] - 1) * 3 + 2]
		);
		shape->addPoint(point, true);
	}
	return shape;
}
