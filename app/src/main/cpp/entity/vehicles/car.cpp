//
// Created by samuel on 28/05/18.
//

#include "car.h"
#include <btBulletDynamicsCommon.h>
#include <android/asset_manager.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../../utils/assets.h"
#include "../../utils/rigidbody.h"


static float wheelRadius = 0.7f;
static float wheelWidth = 0.2f;
static float chassisMass = 150.f;
static float wheelMass = 10.f;

Car::Car(btDynamicsWorld* world, AAssetManager* mgr) {
    init(world, mgr);
}

void Car::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
    btScalar tmp[16];

    // Chassis
    defaultMotionState[0]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
    glm::mat4 modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale[0]);

    glm::mat4 mvMatrix = vMatrix * modelMatrix;
    glm::mat4 mvpMatrix = pMatrix * mvMatrix;

    modelVBOs[0]->draw(mvpMatrix, mvMatrix, lighPos);

    //Wheels
    int idxDebutWheels = 1;
    for (int i = 0; i < 4; i++) {
        int idx = idxDebutWheels + i;
        defaultMotionState[idx]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
        modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale[1]);

        mvMatrix = vMatrix * modelMatrix;
        mvpMatrix = pMatrix * mvMatrix;

        modelVBOs[1]->draw(mvpMatrix, mvMatrix, lighPos);
    }
}

Car::~Car() {
    for (ModelVBO* m : modelVBOs)
        delete m;
}

/**
 * CollisionShape[0] -> chassisShape
 * CollisionShape[1] -> wheel shape
 *
 * rigidBody[0] -> m_carChassis
 * rigidBody[1 - 4] -> pBodyB
 *
 * defaultMotionState[0] -> m_carChassisMotionState
 * defaultMotionState[1 - 4] -> pBodyBMotionState
 *
 * modelVBOs[0] -> chassisShape
 * modelVBOs[1] -> wheels
 *
 * scale[0] -> chassis
 * scale[1] -> wheels
 *
 * @param world
 */
void Car::init(btDynamicsWorld* world, AAssetManager* mgr) {
    std::string cubeObjTxt = getFileText(mgr, "obj/cube.obj");

    float posY = 5.f;

    btTransform tr;
    tr.setIdentity();

    btCollisionShape *chassisShape = new btBoxShape(btVector3(1.f, 0.5f, 2.f));
    collisionShape.push_back(chassisShape);

    tr.setOrigin(btVector3(0, posY + 0.f, 0));

    std::tuple<btRigidBody*, btDefaultMotionState*> tmp = localCreateRigidBody(chassisMass, tr, chassisShape);//chassisShape);
    btRigidBody* m_carChassis = std::get<0>(tmp);
    btDefaultMotionState* m_carChassiMotionState = std::get<1>(tmp);

    modelVBOs.push_back(new ModelVBO(cubeObjTxt, new float[4]{1.f,0.f,0.f,1.f}));
    scale.push_back(glm::vec3(1.f, 0.5f, 2.f));
    rigidBody.push_back(m_carChassis);
    defaultMotionState.push_back(m_carChassiMotionState);

    btCollisionShape* m_wheelShape = new btCylinderShapeX(btVector3(wheelWidth, wheelRadius, wheelRadius));
    collisionShape.push_back(m_wheelShape);

    btVector3 wheelPos[4] = {
            btVector3(btScalar(-1.), btScalar(posY-0.3), btScalar(1.25)),
            btVector3(btScalar(1.), btScalar(posY-0.3), btScalar(1.25)),
            btVector3(btScalar(1.), btScalar(posY-0.3), btScalar(-1.25)),
            btVector3(btScalar(-1.), btScalar(posY-0.3), btScalar(-1.25))
    };

    std::string cylObjText = getFileText(mgr, "obj/cylinderX.obj");
    scale.push_back(glm::vec3(wheelWidth, wheelRadius, wheelRadius));
    modelVBOs.push_back(new ModelVBO(cylObjText, new float[4]{0.f,1.f,0.f,1.f}));

    for (int i = 0; i < 4; i++) {
        // create a Hinge2 joint
        // create two rigid bodies
        // static bodyA (parent) on top:

        btRigidBody *pBodyA = rigidBody[0];//m_chassis;//createRigidBody( 0.0, tr, m_wheelShape);
        pBodyA->setActivationState(DISABLE_DEACTIVATION);
        // dynamic bodyB (child) below it :
        tr.setIdentity();
        tr.setOrigin(wheelPos[i]);

        tmp = localCreateRigidBody(wheelMass, tr, m_wheelShape);

        btRigidBody *pBodyB = std::get<0>(tmp);
        btDefaultMotionState* pBodyBMotionState = std::get<1>(tmp);

        rigidBody.push_back(pBodyB);
        defaultMotionState.push_back(pBodyBMotionState);

        pBodyB->setFriction(1110);
        pBodyB->setActivationState(DISABLE_DEACTIVATION);
        // add some data to build constraint frames
        btVector3 parentAxis(0.f, 1.f, 0.f);
        btVector3 childAxis(1.f, 0.f, 0.f);
        btVector3 anchor = tr.getOrigin();//(0.f, 0.f, 0.f);
        pHinge2.push_back(new btHinge2Constraint(*pBodyA, *pBodyB, anchor, parentAxis,
                                                             childAxis));

        pHinge2[i]->setLowerLimit(float(-M_PI) * 0.5f);
        pHinge2[i]->setUpperLimit(float(M_PI)  * 0.5f);
        // add constraint to world
        world->addConstraint(pHinge2[i], true);
            // draw constraint frames and limits for debugging
        /**
         * Y-axis
         */

        {
            int motorAxis = 3;
            pHinge2[i]->enableMotor(motorAxis, true);
            pHinge2[i]->setMaxMotorForce(motorAxis, 1000);
            pHinge2[i]->setTargetVelocity(motorAxis, -3);
        }

        /*
         * motorAxis 5 -> Y axis
         */
        {
            int motorAxis = 5;
            pHinge2[i]->enableMotor(motorAxis, true);
            pHinge2[i]->setMaxMotorForce(motorAxis, 1e10f);
            pHinge2[i]->setTargetVelocity(motorAxis, 0);
        }


        pHinge2[i]->setDbgDrawSize(btScalar(5.f));
    }
}

void Car::control() {
    /*int wheelIndex = 2;
    m_vehicle->applyEngineForce(gEngineForce,wheelIndex);
    m_vehicle->setBrake(gBreakingForce,wheelIndex);
    wheelIndex = 3;
    m_vehicle->applyEngineForce(gEngineForce,wheelIndex);
    m_vehicle->setBrake(gBreakingForce,wheelIndex);


    wheelIndex = 0;
    m_vehicle->setSteeringValue(gVehicleSteering,wheelIndex);
    wheelIndex = 1;
    m_vehicle->setSteeringValue(gVehicleSteering,wheelIndex);*/
    for (int i = 0; i < 4; i++) {
    }
}

glm::vec3 Car::getCamPos() {
    btScalar tmp[16];

    // Chassis
    defaultMotionState[0]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
    glm::mat4 modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale[0]);

    glm::vec4 pos(0.f,0.f,0.f,1.f);
    pos = modelMatrix * pos;

    return glm::vec3(pos.x, pos.y, pos.z);
}
