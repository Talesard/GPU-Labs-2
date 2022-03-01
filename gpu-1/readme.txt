как компилить:
    запускаем x64 Native Tools Command Prompt for VS 2019
    потом build.bat названиефайла.cpp
    потом мб переделать на *.cpp

    сделал переменную среды src
    теперь можно делать cd %src%

как пофиксить includepath:
            "includePath": [
                "${workspaceFolder}/**",
                "C:\\Users\\veter\\sycl_workspace\\llvm\\build\\include\\sycl",
                "C:\\Users\\veter\\sycl_workspace\\llvm\\build\\include"
            ],
            в интерфейсе vscode без двойных слешей

больше инфы:
    https://intel.github.io/llvm-docs/GetStartedGuide.html#run-simple-dpc-application

clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 %1 -o app.exe