# Ballrain native volunteer client (docs/native-client.md).
#
#   ballrain_core    portable C++17: wire formats, game state, geometry, features, policy, JSON, sha256, npz
#   ballrain_engine  the headless engine session and the env (uses the thin platform layer below)
#   platform/        the only OS-specific code: platform_win.cpp (Windows) / platform_posix.cpp (macOS, Linux, Android)
#
# Windows (from a VS developer prompt, x64):  cmake -S native -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release
# Android: -DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=28
cmake_minimum_required(VERSION 3.16)
# the client version (hello.client_version, the release zip name): the one place to change it
project(ballrain_native VERSION 0.5.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(BALLRAIN_APPS "build the command-line programs" ON)
option(BALLRAIN_TUI "build the terminal UI client ballrain_pool (FTXUI)" ON)
# release builds only (never committed): the gateway prefilled in the setup, the signed release manifest, its key
set(BALLRAIN_DEFAULT_GATEWAY "" CACHE STRING "gateway URL prefilled in the client's setup")
set(BALLRAIN_UPDATE_URL "" CACHE STRING "https URL of the signed release manifest (release.json)")
set(BALLRAIN_RELEASE_PUBKEY "" CACHE STRING "hex ed25519 public key of the release manifest signatures")

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
target_compile_definitions(ballrain_miniz PUBLIC MINIZ_NO_TIME MINIZ_NO_ZLIB_COMPATIBLE_NAMES)  # zip writing: diagnostics
if(NOT MSVC)
    target_compile_options(ballrain_miniz PRIVATE -w)
endif()

add_library(ballrain_zstd STATIC third_party/zstd/zstd.c)
target_include_directories(ballrain_zstd PUBLIC third_party/zstd)
target_compile_definitions(ballrain_zstd PRIVATE ZSTD_DISABLE_ASM=1)
if(NOT MSVC)
    target_compile_options(ballrain_zstd PRIVATE -w)
endif()

# ed25519 (orlp/ed25519, zlib licence): release-manifest signatures. seed.c (the OS random generator) only in the
# release tool that makes keys
add_library(ballrain_ed25519 STATIC
    third_party/ed25519/add_scalar.c third_party/ed25519/fe.c third_party/ed25519/ge.c third_party/ed25519/keypair.c
    third_party/ed25519/sc.c third_party/ed25519/sha512.c third_party/ed25519/sign.c third_party/ed25519/verify.c)
target_include_directories(ballrain_ed25519 PUBLIC third_party/ed25519)
target_compile_definitions(ballrain_ed25519 PRIVATE ED25519_NO_SEED)
if(NOT MSVC)
    target_compile_options(ballrain_ed25519 PRIVATE -w)
endif()

add_library(ballrain_core STATIC
    core/src/util.cpp core/src/json.cpp core/src/sha256.cpp core/src/npz.cpp core/src/wire.cpp core/src/state.cpp
    core/src/geometry.cpp core/src/path.cpp core/src/branch.cpp core/src/features.cpp core/src/policy.cpp core/src/stream.cpp
    core/src/engine_chain.cpp core/src/chunk.cpp core/src/starts.cpp core/src/reward.cpp core/src/fastreach.cpp
    core/src/reach_block.cpp)
target_include_directories(ballrain_core PUBLIC core/include)
target_link_libraries(ballrain_core PRIVATE ballrain_miniz ballrain_zstd)

if(WIN32)
    set(BALLRAIN_PLATFORM_SRC platform/platform_win.cpp)
else()
    set(BALLRAIN_PLATFORM_SRC platform/platform_posix.cpp)
endif()
add_library(ballrain_engine STATIC core/src/session.cpp core/src/env.cpp core/src/task_runner.cpp core/src/pool_client.cpp
    core/src/pool_task.cpp core/src/controller.cpp core/src/update.cpp core/src/diagnostics.cpp
    ${BALLRAIN_PLATFORM_SRC})
target_include_directories(ballrain_engine PUBLIC platform)
target_compile_definitions(ballrain_engine PRIVATE BALLRAIN_CLIENT_VERSION="${PROJECT_VERSION}")
target_link_libraries(ballrain_engine PUBLIC ballrain_core ballrain_ed25519 ballrain_miniz)
if(WIN32)
    target_link_libraries(ballrain_engine PUBLIC ws2_32 winhttp shell32 ole32 advapi32)
    target_compile_definitions(ballrain_engine PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
elseif(NOT ANDROID)
    find_package(Threads REQUIRED)
    target_link_libraries(ballrain_engine PUBLIC Threads::Threads)
    find_package(CURL)  # https on macOS / Linux (without it: http to a loopback gateway only)
    if(CURL_FOUND)
        target_link_libraries(ballrain_engine PUBLIC CURL::libcurl)
        target_compile_definitions(ballrain_engine PRIVATE BALLRAIN_HAVE_CURL=1)
    else()
        message(WARNING "libcurl not found: this build cannot reach an https gateway")
    endif()
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
    add_executable(nc_branch_diff apps/nc_branch_diff.cpp)
    target_link_libraries(nc_branch_diff PRIVATE ballrain_engine)
    add_executable(nc_reach_diff apps/nc_reach_diff.cpp)
    target_link_libraries(nc_reach_diff PRIVATE ballrain_engine)
    # the volunteer client's terminal UI (FTXUI 7.0.3, MIT, vendored in third_party/ftxui)
    if(BALLRAIN_TUI)
        set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        set(FTXUI_QUIET ON CACHE BOOL "" FORCE)
        add_subdirectory(third_party/ftxui EXCLUDE_FROM_ALL)
        add_executable(ballrain_pool apps/ballrain_pool.cpp)
        target_link_libraries(ballrain_pool PRIVATE ballrain_engine ftxui::component ftxui::dom ftxui::screen)
        target_compile_definitions(ballrain_pool PRIVATE
            BALLRAIN_DEFAULT_GATEWAY="${BALLRAIN_DEFAULT_GATEWAY}" BALLRAIN_UPDATE_URL="${BALLRAIN_UPDATE_URL}"
            BALLRAIN_RELEASE_PUBKEY="${BALLRAIN_RELEASE_PUBKEY}")
        if(MSVC)
            target_sources(ballrain_pool PRIVATE platform/utf8.manifest)
        endif()
    endif()
    # release signing (keys from the OS random generator: seed.c only here)
    add_executable(nc_release apps/nc_release.cpp third_party/ed25519/seed.c)
    target_link_libraries(nc_release PRIVATE ballrain_engine)
    if(WIN32)
        target_link_libraries(nc_release PRIVATE advapi32)
    endif()
    if(MSVC)  # UTF-8 argv / environment / narrow file APIs (platform/utf8.manifest)
        foreach(t nc_golden nc_runner nc_client nc_task nc_release)
            target_sources(${t} PRIVATE platform/utf8.manifest)
        endforeach()
    endif()
endif()

# tests: the engine-free self test always; the golden replays when BALLRAIN_GOLDEN_DIR names a directory of
# tools/native/golden.py recordings (<dir>/<name>/stream.bin) and BALLRAIN_MANIFEST / BALLRAIN_PATH_DIR are given
if(BALLRAIN_APPS)
    enable_testing()
    add_executable(nc_selftest tests/selftest.cpp)
    target_link_libraries(nc_selftest PRIVATE ballrain_engine)
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
