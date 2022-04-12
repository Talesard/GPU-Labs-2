mkdir build

set DPCPP_HOME=%USERPROFILE%\sycl_workspace
set PATH=%DPCPP_HOME%\llvm\build\bin;%PATH%
set LIB=%DPCPP_HOME%\llvm\build\lib;%LIB%

clang++ -Wno-unknown-cuda-version -Wno-linker-warnings -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 gpu-1/main.cpp -o build/gpu-1.exe
clang++ -Wno-unknown-cuda-version -Wno-linker-warnings -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 gpu-2/main.cpp -o build/gpu-2.exe
clang++ -Wno-unknown-cuda-version -Wno-linker-warnings -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 gpu-3/main.cpp -o build/gpu-3.exe

build\gpu-1.exe
build\gpu-2.exe 10000 cpu
build\gpu-2.exe 10000 gpu
build\gpu-3.exe 2000 0.00001 200 gpu
build\gpu-3.exe 2000 0.00001 200 cpu

rmdir /q /s build
