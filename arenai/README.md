# ArenAI : C++ part

This is the core modules of ArenAI.

## Installation

Build should work on any decent Linux distribution.

Windows build is now working !

### ArchLinux

Install dependencies with pacman :
```shell
$ sudo pacman -Sy bullet glm glfw
```

Then, download or install LibTorch :

```shell
$ wget https://download.pytorch.org/libtorch/cu132/libtorch-shared-with-deps-2.12.0%2Bcu132.zip
$ unzip ./libtorch-shared-with-deps-2.12.0+cu132.zip -d /opt
```

or from the AUR with your favorite manager (ex : paru) :
```shell
$ # to adapt according your machine (cpu/cuda)
$ paru -Sy libtorch-cuda
```

Finally, build the project :
```shell
$ cd /path/to/ArenAI/arenai/
$ mkdir build && cd build
$ cmake .. && make
```

### Windows

You need first to install Visual Studio C++ (download it from [official website](https://visualstudio.microsoft.com/fr/vs/features/cplusplus/)).
This will add the `vcpkg` executable in the `PATH` which is needed to compile the project and its dependencies.

Then you are ready to build all modules.
Open a PowerShell session :
```powershell
cd C:\Users\MyName\path\to\ArenAI\arenai
.\install_dependencies.ps1
```

This will create `libs` folder with all dependencies.

You can now compile the project's modules :
```powershell
cd C:\Users\MyName\path\to\ArenAI\arenai
.\build_windows.ps1
```

## Note

Any bug, build failed, etc. reports will be really appreciated.
Create your issue if you want to participate !
