cmake_minimum_required(VERSION 3.5)

set(PHYVR_PATH "src/main/cpp")

set(LIBS_PATH "../libs")

set(BULLET_PATH "${LIBS_PATH}/bullet3")
set(GLM_PATH "${LIBS_PATH}/glm")
set(LIBPNG_PATH "${LIBS_PATH}/libpng-android/jni")


##### Bullet3 #####

set(CMAKE_C_FLAGS "-DUSE_PTHREADS -mfpu=neon -mfloat-abi=softfp -pthread -DSCE_PFX_USE_SIMD_VECTORMATH")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -O3")

set(CMAKE_ANDROID_ARM_NEON "true")

list(FILTER CMAKE_C_FLAGS EXCLUDE REGEX "-ffpu=vfp")

file(GLOB BULLET_SRC
		${BULLET_PATH}/src/*.cpp
		${BULLET_PATH}/src/LinearMath/*.cpp
		${BULLET_PATH}/src/Bullet3Common/*.cpp
		${BULLET_PATH}/src/BulletCollision/BroadphaseCollision/*.cpp
		${BULLET_PATH}/src/BulletCollision/CollisionDispatch/*.cpp
		${BULLET_PATH}/src/BulletCollision/CollisionShapes/*.cpp
		${BULLET_PATH}/src/BulletCollision/NarrowPhaseCollision/*.cpp
		${BULLET_PATH}/src/BulletDynamics/ConstraintSolver/*.cpp
		${BULLET_PATH}/src/BulletDynamics/Dynamics/*.cpp
		${BULLET_PATH}/src/BulletDynamics/Featherstone/*.cpp
		${BULLET_PATH}/src/BulletDynamics/MLCPSolvers/*.cpp
		${BULLET_PATH}/src/BulletDynamics/Vehicle/*.cpp
		${BULLET_PATH}/src/BulletDynamics/Character/*.cpp
		${BULLET_PATH}/src/BulletSoftBody/*.cpp
		${BULLET_PATH}/src/BulletInverseDynamics/*.cpp
		${BULLET_PATH}/src/BulletInverseDynamics/details/*.cpp
)

include_directories(${BULLET_PATH}/src)

add_library(bullet3 STATIC ${BULLET_SRC})

unset(${CMAKE_C_FLAGS})
unset(${CMAKE_CXX_FLAGS})
unset(${CMAKE_ANDROID_ARM_NEON})


##### Libpng #####

set(CMAKE_C_FLAGS "-std=gnu89 -O3")

include_directories(${LIBPNG_PATH})

file(GLOB LIBPNG_SRC
	${LIBPNG_PATH}/png.c
	${LIBPNG_PATH}/pngerror.c
	${LIBPNG_PATH}/pngget.c
	${LIBPNG_PATH}/pngmem.c
	${LIBPNG_PATH}/pngpread.c
	${LIBPNG_PATH}/pngread.c
	${LIBPNG_PATH}/pngrio.c
	${LIBPNG_PATH}/pngrtran.c
	${LIBPNG_PATH}/pngrutil.c
	${LIBPNG_PATH}/pngset.c
	${LIBPNG_PATH}/pngtest.c
	${LIBPNG_PATH}/pngtrans.c
	${LIBPNG_PATH}/pngwio.c
	${LIBPNG_PATH}/pngwrite.c
	${LIBPNG_PATH}/pngwtran.c
	${LIBPNG_PATH}/pngwutil.c
)

add_library(libpng STATIC ${LIBPNG_SRC})

target_link_libraries(libpng z)

unset(${CMAKE_C_FLAGS})
unset(${CMAKE_CXX_FLAGS})


##### PhyVR #####

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

file(GLOB_RECURSE PHYVR_SRC "${PHYVR_PATH}/*.cpp" "${PHYVR_PATH}/*.c")

add_library(phyvr SHARED ${PHYVR_SRC})

target_link_libraries(phyvr GLESv3 android log dl bullet3 libpng)

include_directories(${GLM_PATH})