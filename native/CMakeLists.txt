# Ballrain native volunteer client (docs/native-client.md).
#
#   ballrain_core    portable C++17: wire formats, game state, geometry, features, policy, JSON, sha256, npz
#   ballrain_engine  the headless engine session and the env (uses the thin platform layer below)
#   platform/        the only OS-specific code: platform_win.cpp (Windows) / platform_posix.cpp (macOS, Linux, Android)
#
# Windows (from a VS developer prompt, x64):  cmake -S native -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release
# Android: -DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=28
cmake_minimum_required(VERSION 3.16)
project(ballrain_native C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(BALLRAIN_APPS "build the command-line programs" ON)

# one static executable per platform: the static C/C++ runtime on Windows
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# reproducible floating point: no fast-math, no FMA contraction (the verifier compares with tolerances, but fewer
# cross-CPU differences mean fewer boundary cases)
if(MSVC)
    add_compile_options(/fp:precise /utf-8 /W3 /permissive- /D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-ffp-contract=off -fno-fast-math -Wall -Wextra -Wno-unused-parameter)
endif()

add_library(ballrain_miniz STATIC
    third_party/miniz/miniz.c third_party/miniz/miniz_tdef.c third_party/miniz/miniz_tinfl.c
    third_party/miniz/miniz_zip.c)
target_include_directories(ballrain_miniz PUBLIC third_party/miniz)
target_compile_definitions(ballrain_miniz PUBLIC MINIZ_NO_TIME MINIZ_NO_ARCHIVE_WRITING_APIS MINIZ_NO_ZLIB_COMPATIBLE_NAMES)
if(NOT MSVC)
    target_compile_options(ballrain_miniz PRIVATE -w)
endif()

add_library(ballrain_zstd STATIC third_party/zstd/zstd.c)
target_include_directories(ballrain_zstd PUBLIC third_party/zstd)
target_compile_definitions(ballrain_zstd PRIVATE ZSTD_DISABLE_ASM=1)
if(NOT MSVC)
    target_compile_options(ballrain_zstd PRIVATE -w)
endif()

add_library(ballrain_core STATIC
    core/src/util.cpp core/src/json.cpp core/src/sha256.cpp core/src/npz.cpp core/src/wire.cpp core/src/state.cpp
    core/src/geometry.cpp core/src/path.cpp core/src/features.cpp core/src/policy.cpp core/src/stream.cpp
    core/src/engine_chain.cpp core/src/chunk.cpp core/src/starts.cpp core/src/reward.cpp)
target_include_directories(ballrain_core PUBLIC core/include)
target_link_libraries(ballrain_core PRIVATE ballrain_miniz ballrain_zstd)

if(WIN32)
    set(BALLRAIN_PLATFORM_SRC platform/platform_win.cpp)
else()
    set(BALLRAIN_PLATFORM_SRC platform/platform_posix.cpp)
endif()
add_library(ballrain_engine STATIC core/src/session.cpp core/src/env.cpp core/src/task_runner.cpp core/src/pool_client.cpp
    core/src/pool_task.cpp core/src/controller.cpp
    ${BALLRAIN_PLATFORM_SRC})
target_include_directories(ballrain_engine PUBLIC platform)
target_link_libraries(ballrain_engine PUBLIC ballrain_core)
if(WIN32)
    target_link_libraries(ballrain_engine PUBLIC ws2_32 winhttp shell32 ole32)
    target_compile_definitions(ballrain_engine PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
elseif(NOT ANDROID)
    find_package(Threads REQUIRED)
    target_link_libraries(ballrain_engine PUBLIC Threads::Threads)
endif()

if(BALLRAIN_APPS)
    add_executable(nc_golden apps/nc_golden.cpp)
    target_link_libraries(nc_golden PRIVATE ballrain_engine)
    add_executable(nc_runner apps/nc_runner.cpp)
    target_link_libraries(nc_runner PRIVATE ballrain_engine)
    add_executable(nc_client apps/nc_client.cpp)
    target_link_libraries(nc_client PRIVATE ballrain_engine)
    add_executable(nc_task apps/nc_task.cpp)
    target_link_libraries(nc_task PRIVATE ballrain_engine)
endif()

# tests: the engine-free self test always; the golden replays when BALLRAIN_GOLDEN_DIR names a directory of
# tools/native/golden.py recordings (<dir>/<name>/stream.bin) and BALLRAIN_MANIFEST / BALLRAIN_PATH_DIR are given
if(BALLRAIN_APPS)
    enable_testing()
    add_executable(nc_selftest tests/selftest.cpp)
    target_link_libraries(nc_selftest PRIVATE ballrain_core)
    add_test(NAME selftest COMMAND nc_selftest)
    if(BALLRAIN_GOLDEN_DIR AND BALLRAIN_MANIFEST AND BALLRAIN_PATH_DIR)
        file(GLOB _goldens LIST_DIRECTORIES true "${BALLRAIN_GOLDEN_DIR}/*")
        foreach(g ${_goldens})
            if(EXISTS "${g}/stream.bin")
                get_filename_component(n "${g}" NAME)
                add_test(NAME golden_${n} COMMAND nc_golden --manifest "${BALLRAIN_MANIFEST}" --stream "${g}/stream.bin"
                         --path-dir "${BALLRAIN_PATH_DIR}")
            endif()
        endforeach()
    endif()
endif()
