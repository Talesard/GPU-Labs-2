# DPC++ (SYCL) Labs
- [DPC++ (SYCL) Labs](#dpc-sycl-labs)
  - [Install dependencies](#install-dependencies)
  - [Build a program](#build-a-program)
  - [Visual Studio Code settings](#visual-studio-code-settings)
  - [List of labs](#list-of-labs)

## Install dependencies
- Cuda (https://developer.nvidia.com/cuda-downloads)
- Intel OpenCL runtime for CPU (https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html)
- Build LLVM with --cuda (https://intel.github.io/llvm-docs/GetStartedGuide.html)

## Build a program
- Use the x64 native tools MS (for me ntvs shortcut)
- ```build.bat filename.cpp```
- ```app.exe```

## Visual Studio Code settings
- Install C/C++ Extension Pack
- Fix .vscode/c_cpp_properties.json:
```
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**",
                "C:\\Users\\veter\\sycl_workspace\\llvm\\build\\include\\sycl",
                "C:\\Users\\veter\\sycl_workspace\\llvm\\build\\include"
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "_UNICODE"
            ],
            "windowsSdkVersion": "10.0.19041.0",
            "compilerPath": "C:/Users/veter/sycl_workspace/llvm/build/bin/clang++.exe",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "windows-clang-x64"
        }
    ],
    "version": 4
}
```

## List of labs
- Hello, world!
- Double integrals computatuion with Riemann sums
- Solving SLE by the Jacobi method
