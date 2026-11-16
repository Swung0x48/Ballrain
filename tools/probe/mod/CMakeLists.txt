cmake_minimum_required(VERSION 3.14)

project(BallrainProbe LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "x86 toolchain required")
endif ()

set(BML_DIR "" CACHE PATH "")
set(VirtoolsSDK_DIR "" CACHE PATH "")
find_package(BML REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../BallrainIO/3rdparty ${CMAKE_BINARY_DIR}/3rdparty)

add_library(BallrainProbe SHARED
        src/Probe.h
        src/Probe.cpp
        src/ProbeDump.cpp
        src/Hook.h
        src/TASHook.h
)

target_link_libraries(BallrainProbe PRIVATE BML minhook bcrypt)
set_target_properties(BallrainProbe PROPERTIES OUTPUT_NAME "BallrainIOProbe" SUFFIX ".bmodp")
