cmake_minimum_required(VERSION 3.10)

project(BallrainIO)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(3rdparty)

set(BML_DIR "${CMAKE_CURRENT_LIST_DIR}/3rdparty/BMLPlus/lib/cmake/BML")
message(STATUS "BML_DIR=${BML_DIR}")
find_package(BML REQUIRED)

add_library(BallrainIO SHARED
			src/BallrainIO.h
			src/BallrainIO.cpp
			src/InputSystem.h
			src/InputSystem.cpp
			src/TimeSystem.h
			src/TimeSystem.cpp
			src/Hook.h
			src/TASHook.cpp
			src/TASHook.h
			src/TCPClient.cpp
			src/TCPClient.h
)

target_include_directories(BallrainIO PRIVATE ${BML_INCLUDE_DIRS})
target_link_libraries(BallrainIO
						PUBLIC
						BML
						PRIVATE
						minhook)
set_target_properties(BallrainIO PROPERTIES OUTPUT_NAME "BallrainIO")
set_target_properties(BallrainIO PROPERTIES SUFFIX ".bmodp")