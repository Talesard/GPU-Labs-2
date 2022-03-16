:: запускать из x64 Native Tools Command Prompt for VS 2019
:: 1 аргумент - файл
@echo off
set DPCPP_HOME=%USERPROFILE%\sycl_workspace
set PATH=%DPCPP_HOME%\llvm\build\bin;%PATH%
set LIB=%DPCPP_HOME%\llvm\build\lib;%LIB%

@echo on
clang++ -Wno-unknown-cuda-version -Wno-linker-warnings -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 %1 -o app.exe
