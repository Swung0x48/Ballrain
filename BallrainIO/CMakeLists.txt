cmake_minimum_required(VERSION 3.10)

project(BallrainIO)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BML REQUIRED)

add_library(BallrainIO SHARED
			src/BallrainIO.h
			src/BallrainIO.cpp
			src/InputSystem.h
			src/InputSystem.cpp
			src/TimeSystem.h
			src/TimeSystem.cpp)

target_include_directories(BallrainIO PRIVATE ${BML_INCLUDE_DIRS})
target_link_libraries(BallrainIO PRIVATE BML)
set_target_properties(BallrainIO PROPERTIES OUTPUT_NAME "BallrainIO")
set_target_properties(BallrainIO PROPERTIES SUFFIX ".bmodp")