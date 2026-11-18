@echo off
rem Windows x64 build of the native client (static executables in build\native\).
rem   native\build_win.cmd [Release|Debug]
rem VCVARS: vcvars64.bat of Visual Studio (default: VS 18 Community, then VS 2022 Community / BuildTools).
setlocal
set CFG=%1
if "%CFG%"=="" set CFG=Release
if "%VCVARS%"=="" if exist "%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if "%VCVARS%"=="" if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if "%VCVARS%"=="" if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if "%VCVARS%"=="" (echo vcvars64.bat not found; set VCVARS & exit /b 1)
call "%VCVARS%" >nul || exit /b 1
set HERE=%~dp0
rem BALLRAIN_BUILD_DIR: another build directory (default build\native-<cfg>); BALLRAIN_CMAKE_ARGS: extra -D options
if "%BALLRAIN_BUILD_DIR%"=="" set "BALLRAIN_BUILD_DIR=%HERE%..\build\native-%CFG%"
cmake -S "%HERE%." -B "%BALLRAIN_BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=%CFG% %BALLRAIN_CMAKE_ARGS% || exit /b 1
cmake --build "%BALLRAIN_BUILD_DIR%" || exit /b 1
