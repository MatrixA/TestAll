cmake_minimum_required(VERSION 2.8)
project(test_opengl)

set(CMAKE_VERBOSE_MAKEFILE OFF)

find_package(OpenCV 4)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Ceres REQUIRED)
include_directories( ${CERES_INCLUDE_DIRS})

find_package(PkgConfig)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)
include_directories(${YAMLCPP_INCLUDE_DIRS})
message(INFO ${YAMLCPP_INCLUDE_DIRS})
link_directories(${YAMLCPP_LIBRARY_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# glfw和glew的头文件目录，这是我的路径，你可以换成你自己的



aux_source_directory("./" SRCS)

set(CMAKE_CXX_FLAGS "-g -std=c++14 -Wformat")


add_executable(test_opengl ${SRCS} ${HDRS})


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${CERES_LIBRARIES}
${Pangolin_LIBRARIES}
${YAMLCPP_LIBRARIES}
${PCL_LIBRARIES}
)
