cmake_minimum_required(VERSION 3.10)

project(BallrainEnv)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BML REQUIRED)

add_library(BallrainEnv SHARED
			src/BallrainEnv.h
			src/BallrainEnv.cpp)

target_include_directories(BallrainEnv PRIVATE ${BML_INCLUDE_DIRS})
target_link_libraries(BallrainEnv PRIVATE BML)
set_target_properties(BallrainEnv PROPERTIES OUTPUT_NAME "BallrainEnv")
set_target_properties(BallrainEnv PROPERTIES SUFFIX ".bmodp")