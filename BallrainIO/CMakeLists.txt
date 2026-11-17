cmake_minimum_required(VERSION 3.14)

project(BallrainIO LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "BallrainIO is a mod for the 32-bit game: configure with an x86 toolchain")
endif ()

# Machine paths come from the cache (e.g. a gitignored build .bat or CMakeUserPresets.json):
#   BML_DIR          = <BML+ 0.3.12 SDK>/lib/cmake/BML        (Release only)
#   VirtoolsSDK_DIR  = <Virtools SDK 2.1>/CMake               (needed by BMLConfig's find_dependency)
set(BML_DIR "" CACHE PATH "Directory holding BMLConfig.cmake (BML+ 0.3.12 SDK)")
set(VirtoolsSDK_DIR "" CACHE PATH "Directory holding VirtoolsSDKConfig.cmake")
find_package(BML REQUIRED)

add_subdirectory(3rdparty)

add_library(BallrainIO SHARED
        src/BallrainIO.h
        src/BallrainIO.cpp
        src/Hook.h
        src/TASHook.h
        src/Agent.cpp
        src/Capture.cpp
        src/Ivp2002.h
        src/Wire.h
)

target_link_libraries(BallrainIO PRIVATE BML minhook bcrypt)
set_target_properties(BallrainIO PROPERTIES OUTPUT_NAME "BallrainIO" SUFFIX ".bmodp")
