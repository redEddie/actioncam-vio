cmake_minimum_required(VERSION 3.10)
project(umi_offline_slam)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DCOMPILEDWITHC11)   # selects std::chrono::steady_clock path in ORB-SLAM3 headers

# Path to the (already built) ORB-SLAM3 tree. Override with -DORB_SLAM3_ROOT=...
if(NOT DEFINED ORB_SLAM3_ROOT)
  set(ORB_SLAM3_ROOT "$ENV{HOME}/rs_slam/ORB_SLAM3")
endif()
message(STATUS "ORB_SLAM3_ROOT = ${ORB_SLAM3_ROOT}")

find_package(OpenCV 4.4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)        # found via CMAKE_PREFIX_PATH=<install prefix>
find_package(realsense2 REQUIRED)

include_directories(
  ${ORB_SLAM3_ROOT}
  ${ORB_SLAM3_ROOT}/include
  ${ORB_SLAM3_ROOT}/include/CameraModels
  ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
)

add_executable(stereo_inertial_db3 stereo_inertial_db3.cc)
target_link_libraries(stereo_inertial_db3
  ${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  ${realsense2_LIBRARY}
  -lboost_serialization
  -lcrypto
)
