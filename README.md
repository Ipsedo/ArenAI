# ArenAI

A battle-royal game with trained agent which controls tanks in realistic physic world.

## Description

Each tank receives the rendered frame of its camera as input, and the agent is trained to fire and hit enemies.

When the agent is trained (with the PPO or SAC algorithm), you can fight it through the tanks it handles.

## Installation

First you need to clone the repo :
```bash
$ git clone https://github.com/Ipsedo/ArenAI.git
```

Build should work on any decent Linux distribution.

Windows build is now working !

### ArchLinux

Install dependencies with pacman :
```shell
$ sudo pacman -Sy glm glfw vulkan-devel vulkan-headers glslang gtest
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
$ cd /path/to/ArenAI/
$ mkdir build && cd build
$ cmake .. && make -j $(nproc)
```

### Windows

You need first to install Visual Studio C++ (download it from [official website](https://visualstudio.microsoft.com/fr/vs/features/cplusplus/)).
This will add the `vcpkg` executable in the `PATH` which is needed to compile the project and its dependencies.

Then you are ready to build all modules.
Open a PowerShell session :
```powershell
cd C:\Users\MyName\path\to\ArenAI\
.\install_dependencies.ps1
```

This will create `libs` folder with all dependencies.

You can now compile the project's modules :
```powershell
cd C:\Users\MyName\path\to\ArenAI\
.\build_windows.ps1
```

## Known issues

Any bug, build failed, etc. reports will be really appreciated.
Create your issue if you want to participate !

### GNOME system monitor extension

The `system-monitor` extension is causing freeze periodically. The fix is to disable it when playing :
```bash
# on ArchLinux
gnome-extensions disable system-monitor@gnome-shell-extensions.gcampax.github.com
```
