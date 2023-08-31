cmake_minimum_required(VERSION 3.0.2)
project(implicit_lidar_network)


SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_VERSION 11)

if(catkin_FOUND)
    find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        sensor_msgs
        pcl_ros
    )
endif()

FIND_PACKAGE(Eigen3 REQUIRED)

INCLUDE(FindOpenMP)
if(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
ADD_DEFINITIONS(-DOPENMP)

if(catkin_FOUND)
    catkin_package()
endif()
    include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
    )

add_subdirectory(extern/pybind11)
pybind11_add_module(voxelizer SHARED src/voxelizer.cpp)
set_target_properties(voxelizer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python_src
)
