@echo off
rem Build the x86 no-focus launcher into <repo>\runtime\nofocus (static CRT, no other dependencies).
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS (echo Visual Studio C++ tools not found & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvarsamd64_x86.bat" >nul 2>nul || exit /b 1
set "OUT=%~dp0..\..\runtime\nofocus"
if not exist "%OUT%" mkdir "%OUT%"
cd /d "%OUT%"
cl /nologo /O2 /W3 /LD /MT "%~dp0nofocus.c" /link user32.lib /OUT:nofocus.dll || exit /b 1
cl /nologo /O2 /W3 /MT "%~dp0nofocus_launch.c" /link /OUT:nofocus_launch.exe || exit /b 1
del /q *.obj *.exp *.lib 2>nul
echo NOFOCUS-BUILD-OK
