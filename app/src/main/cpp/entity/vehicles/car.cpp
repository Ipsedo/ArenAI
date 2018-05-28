//
// Created by samuel on 28/05/18.
//

#include "car.h"
#include <btBulletDynamicsCommon.h>

Car::Car() {

}

void Car::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {

}

Car::~Car() {

}

void Car::init() {


    btCollisionShape *groundShape = new btBoxShape(btVector3(50, 3, 50));
    collisionShape.push_back(groundShape);
    //m_collisionConfiguration = new btDefaultCollisionConfiguration();
    //m_dispatcher = new btCollisionDispatcher(m_collisionConfiguration);
    btVector3 worldMin(-1000, -1000, -1000);
    btVector3 worldMax(1000, 1000, 1000);
    btTransform tr;
    tr.setIdentity();

    tr.setOrigin(btVector3(0, -3, 0));



    //create ground object

    btCollisionShape *chassisShape = new btBoxShape(btVector3(1.f, 0.5f, 2.f));
    collisionShape.push_back(chassisShape);

    btCompoundShape *compound = new btCompoundShape();
    collisionShape.push_back(compound);
    btTransform localTrans;
    localTrans.setIdentity();

    //localTrans effectively shifts the center of mass with respect to the chassis
    localTrans.setOrigin(btVector3(0, 1, 0));

    compound->addChildShape(localTrans, chassisShape);

    {
        btCollisionShape *suppShape = new btBoxShape(btVector3(0.5f, 0.1f, 0.5f));
        btTransform suppLocalTrans;
        suppLocalTrans.setIdentity();

        //localTrans effectively shifts the center of mass with respect to the chassis
        suppLocalTrans.setOrigin(btVector3(0, 1.0, 2.5));
        compound->addChildShape(suppLocalTrans, suppShape);
    }

    tr.setOrigin(btVector3(0, 0.f, 0));
/*
    btScalar chassisMass = 800;
    m_carChassis = localCreateRigidBody(chassisMass, tr, compound);//chassisShape);
    //m_carChassis->setDamping(0.2,0.2);

    //m_wheelShape = new btCylinderShapeX(btVector3(wheelWidth,wheelRadius,wheelRadius));
    m_wheelShape = new btCylinderShapeX(btVector3(wheelWidth, wheelRadius, wheelRadius));


//const float position[4]={0,10,10,0};
//const float quaternion[4]={0,0,0,1};
//const float color[4]={0,1,0,1};
//const float scaling[4] = {1,1,1,1};

    btVector3 wheelPos[4] = {
            btVector3(btScalar(-1.), btScalar(-0.25), btScalar(1.25)),
            btVector3(btScalar(1.), btScalar(-0.25), btScalar(1.25)),
            btVector3(btScalar(1.), btScalar(-0.25), btScalar(-1.25)),
            btVector3(btScalar(-1.), btScalar(-0.25), btScalar(-1.25))
    };


    for (
            int i = 0;
            i < 4; i++) {
// create a Hinge2 joint
// create two rigid bodies
// static bodyA (parent) on top:


        btRigidBody *pBodyA = this->m_carChassis;//m_chassis;//createRigidBody( 0.0, tr, m_wheelShape);
        pBodyA->setActivationState(DISABLE_DEACTIVATION);
// dynamic bodyB (child) below it :
        btTransform tr;
        tr.

                setIdentity();

        tr.
                setOrigin(wheelPos[i]);

        btRigidBody *pBodyB = createRigidBody(10.0, tr, m_wheelShape);
        pBodyB->setFriction(1110);
        pBodyB->setActivationState(DISABLE_DEACTIVATION);
// add some data to build constraint frames
        btVector3 parentAxis(0.f, 1.f, 0.f);
        btVector3 childAxis(1.f, 0.f, 0.f);
        btVector3 anchor = tr.getOrigin();//(0.f, 0.f, 0.f);
        btHinge2Constraint *pHinge2 = new btHinge2Constraint(*pBodyA, *pBodyB, anchor, parentAxis,
                                                             childAxis);

//m_guiHelper->get2dCanvasInterface();


        pHinge2->setLowerLimit(-SIMD_HALF_PI * 0.5f);
        pHinge2->
                setUpperLimit(SIMD_HALF_PI * 0.5f);
// add constraint to world
        m_dynamicsWorld->
                addConstraint(pHinge2,
                              true);
// draw constraint frames and limits for debugging
        {
            int motorAxis = 3;
            pHinge2->
                    enableMotor(motorAxis,
                                true);
            pHinge2->
                    setMaxMotorForce(motorAxis,
                                     1000);
            pHinge2->
                    setTargetVelocity(motorAxis,
                                      -1);
        }

        {
            int motorAxis = 5;
            pHinge2->
                    enableMotor(motorAxis,
                                true);
            pHinge2->
                    setMaxMotorForce(motorAxis,
                                     1000);
            pHinge2->
                    setTargetVelocity(motorAxis,
                                      0);
        }

        pHinge2->
                setDbgDrawSize(btScalar(5.f));
    }


    {
        btCollisionShape *liftShape = new btBoxShape(btVector3(0.5f, 2.0f, 0.05f));
        m_collisionShapes.
                push_back(liftShape);
        btTransform liftTrans;
        m_liftStartPos = btVector3(0.0f, 2.5f, 3.05f);
        liftTrans.

                setIdentity();

        liftTrans.
                setOrigin(m_liftStartPos);
        m_liftBody = localCreateRigidBody(10, liftTrans, liftShape);

        btTransform localA, localB;
        localA.

                setIdentity();

        localB.

                setIdentity();

        localA.

                        getBasis()

                .setEulerZYX(0, M_PI_2, 0);
        localA.
                setOrigin(btVector3(0.0, 1.0, 3.05));
        localB.

                        getBasis()

                .setEulerZYX(0, M_PI_2, 0);
        localB.
                setOrigin(btVector3(0.0, -1.5, -0.05));
        m_liftHinge = new btHingeConstraint(*m_carChassis, *m_liftBody, localA, localB);
//		m_liftHinge->setLimit(-LIFT_EPS, LIFT_EPS);
        m_liftHinge->setLimit(0.0f, 0.0f);
        m_dynamicsWorld->
                addConstraint(m_liftHinge,
                              true);

        btCollisionShape *forkShapeA = new btBoxShape(btVector3(1.0f, 0.1f, 0.1f));
        m_collisionShapes.
                push_back(forkShapeA);
        btCompoundShape *forkCompound = new btCompoundShape();
        m_collisionShapes.
                push_back(forkCompound);
        btTransform forkLocalTrans;
        forkLocalTrans.

                setIdentity();

        forkCompound->
                addChildShape(forkLocalTrans, forkShapeA
        );

        btCollisionShape *forkShapeB = new btBoxShape(btVector3(0.1f, 0.02f, 0.6f));
        m_collisionShapes.
                push_back(forkShapeB);
        forkLocalTrans.

                setIdentity();

        forkLocalTrans.
                setOrigin(btVector3(-0.9f, -0.08f, 0.7f));
        forkCompound->
                addChildShape(forkLocalTrans, forkShapeB
        );

        btCollisionShape *forkShapeC = new btBoxShape(btVector3(0.1f, 0.02f, 0.6f));
        m_collisionShapes.
                push_back(forkShapeC);
        forkLocalTrans.

                setIdentity();

        forkLocalTrans.
                setOrigin(btVector3(0.9f, -0.08f, 0.7f));
        forkCompound->
                addChildShape(forkLocalTrans, forkShapeC
        );

        btTransform forkTrans;
        m_forkStartPos = btVector3(0.0f, 0.6f, 3.2f);
        forkTrans.

                setIdentity();

        forkTrans.
                setOrigin(m_forkStartPos);
        m_forkBody = localCreateRigidBody(5, forkTrans, forkCompound);

        localA.

                setIdentity();

        localB.

                setIdentity();

        localA.

                        getBasis()

                .setEulerZYX(0, 0, M_PI_2);
        localA.
                setOrigin(btVector3(0.0f, -1.9f, 0.05f));
        localB.

                        getBasis()

                .setEulerZYX(0, 0, M_PI_2);
        localB.
                setOrigin(btVector3(0.0, 0.0, -0.1));
        m_forkSlider = new btSliderConstraint(*m_liftBody, *m_forkBody, localA, localB, true);
        m_forkSlider->setLowerLinLimit(0.1f);
        m_forkSlider->setUpperLinLimit(0.1f);
//		m_forkSlider->setLowerAngLimit(-LIFT_EPS);
//		m_forkSlider->setUpperAngLimit(LIFT_EPS);
        m_forkSlider->setLowerAngLimit(0.0f);
        m_forkSlider->setUpperAngLimit(0.0f);
        m_dynamicsWorld->
                addConstraint(m_forkSlider,
                              true);


        btCompoundShape *loadCompound = new btCompoundShape();
        m_collisionShapes.
                push_back(loadCompound);
        btCollisionShape *loadShapeA = new btBoxShape(btVector3(2.0f, 0.5f, 0.5f));
        m_collisionShapes.
                push_back(loadShapeA);
        btTransform loadTrans;
        loadTrans.

                setIdentity();

        loadCompound->
                addChildShape(loadTrans, loadShapeA
        );
        btCollisionShape *loadShapeB = new btBoxShape(btVector3(0.1f, 1.0f, 1.0f));
        m_collisionShapes.
                push_back(loadShapeB);
        loadTrans.

                setIdentity();

        loadTrans.
                setOrigin(btVector3(2.1f, 0.0f, 0.0f));
        loadCompound->
                addChildShape(loadTrans, loadShapeB
        );
        btCollisionShape *loadShapeC = new btBoxShape(btVector3(0.1f, 1.0f, 1.0f));
        m_collisionShapes.
                push_back(loadShapeC);
        loadTrans.

                setIdentity();

        loadTrans.
                setOrigin(btVector3(-2.1f, 0.0f, 0.0f));
        loadCompound->
                addChildShape(loadTrans, loadShapeC
        );
        loadTrans.

                setIdentity();

        m_loadStartPos = btVector3(0.0f, 3.5f, 7.0f);
        loadTrans.
                setOrigin(m_loadStartPos);
        m_loadBody = localCreateRigidBody(loadMass, loadTrans, loadCompound);
    }
*/
}
