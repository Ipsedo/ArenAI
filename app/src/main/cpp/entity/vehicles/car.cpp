//
// Created by samuel on 28/05/18.
//

#include "car.h"
#include <btBulletDynamicsCommon.h>
#include <android/asset_manager.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <android/log.h>
#include "../../utils/assets.h"


static float wheelRadius = 0.5f;
static float wheelWidth = 0.4f;
static btScalar loadMass = 350.f;

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
    defaultMotionState[5]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
    modelMatrix = glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), scale[2]);

    mvMatrix = vMatrix * modelMatrix;
    mvpMatrix = pMatrix * mvMatrix;

    modelVBOs[2]->draw(mvpMatrix, mvMatrix, lighPos);
}

Car::~Car() {
    for (ModelVBO* m : modelVBOs)
        delete m;
}

std::tuple<btRigidBody*, btDefaultMotionState*> localCreateRigidBody(btScalar mass, const btTransform& startTransform, btCollisionShape* shape)
{
    btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

    //rigidbody is dynamic if and only if mass is non zero, otherwise static
    bool isDynamic = (mass != 0.f);

    btVector3 localInertia(0,0,0);
    if (isDynamic)
        shape->calculateLocalInertia(mass,localInertia);

    btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);

    btRigidBody::btRigidBodyConstructionInfo cInfo(mass,myMotionState,shape,localInertia);

    return std::tuple<btRigidBody*, btDefaultMotionState*>(new btRigidBody(cInfo), myMotionState);
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
#if 1

    std::string cubeObjTxt = getFileText(mgr, "obj/cube.obj");

    btTransform tr;
    tr.setIdentity();

    btCollisionShape *chassisShape = new btBoxShape(btVector3(1.f, 0.5f, 2.f));
    collisionShape.push_back(chassisShape);

    tr.setOrigin(btVector3(0, 0.f, 0));

    btScalar chassisMass = 800;
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
            btVector3(btScalar(-1.), btScalar(-0.25), btScalar(1.25)),
            btVector3(btScalar(1.), btScalar(-0.25), btScalar(1.25)),
            btVector3(btScalar(1.), btScalar(-0.25), btScalar(-1.25)),
            btVector3(btScalar(-1.), btScalar(-0.25), btScalar(-1.25))
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

        //btRigidBody *pBodyB = createRigidBody(10.0, tr, m_wheelShape);
        tmp = localCreateRigidBody(10.0, tr, m_wheelShape);

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

        pHinge2[i]->setLowerLimit(-MATH_PI * 0.5f);
        pHinge2[i]->setUpperLimit(MATH_PI * 0.5f);
        // add constraint to world
        world->addConstraint(pHinge2[i], true);
            // draw constraint frames and limits for debugging
        /*{
            int motorAxis = 3;
            pHinge2[i]->enableMotor(motorAxis, true);
            pHinge2[i]->setMaxMotorForce(motorAxis, 1000);
            pHinge2[i]->setTargetVelocity(motorAxis, -1);
        }

        {
            int motorAxis = 5;
            pHinge2[i]->enableMotor(motorAxis, true);
            pHinge2[i]->setMaxMotorForce(motorAxis, 1000);
            pHinge2[i]->setTargetVelocity(motorAxis, 0);
        }*/

        pHinge2[i]->setDbgDrawSize(btScalar(5.f));
    }
#endif
#if 0
    // This function is called only from the main program when initially creating the vehicle, not on scene load
    /*ResourceCache* cache = GetSubsystem<ResourceCache>();

    StaticModel* hullObject = node_->CreateComponent<StaticModel>();
    hullBody_ = node_->CreateComponent<RigidBody>();
    CollisionShape* hullColShape = node_->CreateComponent<CollisionShape>();

    hullBody_->SetMass(800.0f);
    hullBody_->SetLinearDamping(0.2f); // Some air resistance
    hullBody_->SetAngularDamping(0.5f);
    hullBody_->SetCollisionLayer(1);*/

    std::string cubeObjTxt = getFileText(mgr, "obj/cube.obj");

    btTransform tr;
    tr.setIdentity();

    btCollisionShape *chassisShape = new btBoxShape(btVector3(1.f, 0.5f, 2.f));
    collisionShape.push_back(chassisShape);

    tr.setOrigin(btVector3(0, 0.f, 0));

    btScalar chassisMass = 800;
    std::tuple<btRigidBody*, btDefaultMotionState*> tmp = localCreateRigidBody(chassisMass, tr, chassisShape);//chassisShape);
    btRigidBody* m_carChassis = std::get<0>(tmp);
    btDefaultMotionState* m_carChassiMotionState = std::get<1>(tmp);

    modelVBOs.push_back(new ModelVBO(cubeObjTxt, new float[4]{1.f,0.f,0.f,1.f}));
    scale.push_back(glm::vec3(1.f, 0.5f, 2.f));
    rigidBody.push_back(m_carChassis);
    defaultMotionState.push_back(m_carChassiMotionState);

    int rightIndex = 0;
    int upIndex = 1;
    int forwardIndex = 2;
    /*Scene* scene = GetScene();
    PhysicsWorld *pPhysWorld = scene->GetComponent<PhysicsWorld>();
    btDynamicsWorld *pbtDynWorld = (btDynamicsWorld*)pPhysWorld->GetWorld();*/

    m_vehicleRayCaster = new btDefaultVehicleRaycaster( pbtDynWorld );
    m_vehicle = new btRaycastVehicle( m_tuning, hullBody_->GetBody(), m_vehicleRayCaster );
    world->addVehicle( m_vehicle );

    m_vehicle->setCoordinateSystem( rightIndex, upIndex, forwardIndex );

    node_->SetScale( Vector3(1.5f, 1.0f, 3.5f) );
    Vector3 v3BoxExtents = Vector3::ONE;//Vector3(1.5f, 1.0f, 3.0f);
    hullColShape->SetBox( v3BoxExtents );

    hullObject->SetModel(cache->GetResource<Model>("Models/Box.mdl"));
    hullObject->SetMaterial(cache->GetResource<Material>("Materials/Stone.xml"));
    hullObject->SetCastShadows(true);

    float connectionHeight = -0.4f;//1.2f;
    bool isFrontWheel=true;
    btVector3 wheelDirectionCS0(0,-1,0);
    btVector3 wheelAxleCS(-1,0,0);

    btVector3 connectionPointCS0(CUBE_HALF_EXTENTS-(0.3f*m_fwheelWidth),connectionHeight,2*CUBE_HALF_EXTENTS-m_fwheelRadius);
    m_vehicle->addWheel(connectionPointCS0,wheelDirectionCS0,wheelAxleCS,m_fsuspensionRestLength,m_fwheelRadius,m_tuning,isFrontWheel);

    connectionPointCS0 = btVector3(-CUBE_HALF_EXTENTS+(0.3f*m_fwheelWidth),connectionHeight,2*CUBE_HALF_EXTENTS-m_fwheelRadius);
    m_vehicle->addWheel(connectionPointCS0,wheelDirectionCS0,wheelAxleCS,m_fsuspensionRestLength,m_fwheelRadius,m_tuning,isFrontWheel);

    isFrontWheel = false;
    connectionPointCS0 = btVector3(-CUBE_HALF_EXTENTS+(0.3f*m_fwheelWidth),connectionHeight,-2*CUBE_HALF_EXTENTS+m_fwheelRadius);
    m_vehicle->addWheel(connectionPointCS0,wheelDirectionCS0,wheelAxleCS,m_fsuspensionRestLength,m_fwheelRadius,m_tuning,isFrontWheel);

    connectionPointCS0 = btVector3(CUBE_HALF_EXTENTS-(0.3f*m_fwheelWidth),connectionHeight,-2*CUBE_HALF_EXTENTS+m_fwheelRadius);
    m_vehicle->addWheel(connectionPointCS0,wheelDirectionCS0,wheelAxleCS,m_fsuspensionRestLength,m_fwheelRadius,m_tuning,isFrontWheel);

    for ( int i = 0; i < m_vehicle->getNumWheels(); i++ )
    {
        btWheelInfo& wheel = m_vehicle->getWheelInfo( i );
        wheel.m_suspensionStiffness = m_fsuspensionStiffness;
        wheel.m_wheelsDampingRelaxation = m_fsuspensionDamping;
        wheel.m_wheelsDampingCompression = m_fsuspensionCompression;
        wheel.m_frictionSlip = m_fwheelFriction;
        wheel.m_rollInfluence = m_frollInfluence;
    }

    if ( m_vehicle )
    {
        m_vehicle->resetSuspension();

        for ( int i = 0; i < m_vehicle->getNumWheels(); i++ )
        {
            //synchronize the wheels with the (interpolated) chassis worldtransform
            m_vehicle->updateWheelTransform(i,true);

            btTransform transform = m_vehicle->getWheelTransformWS( i );
            Vector3 v3Origin = ToVector3( transform.getOrigin() );
            Quaternion qRot = ToQuaternion( transform.getRotation() );

            // create wheel node
            Node *wheelNode = GetScene()->CreateChild();
            m_vpNodeWheel.Push( wheelNode );

            wheelNode->SetPosition( v3Origin );
            btWheelInfo whInfo = m_vehicle->getWheelInfo( i );
            Vector3 v3PosLS = ToVector3( whInfo.m_chassisConnectionPointCS );

            wheelNode->SetRotation( v3PosLS.x_ >= 0.0 ? Quaternion(0.0f, 0.0f, -90.0f) : Quaternion(0.0f, 0.0f, 90.0f) );
            wheelNode->SetScale(Vector3(1.0f, 0.65f, 1.0f));

            StaticModel *pWheel = wheelNode->CreateComponent<StaticModel>();
            pWheel->SetModel(cache->GetResource<Model>("Models/Cylinder.mdl"));
            pWheel->SetMaterial(cache->GetResource<Material>("Materials/Stone.xml"));
            pWheel->SetCastShadows(true);
        }
    }
#endif
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
}
