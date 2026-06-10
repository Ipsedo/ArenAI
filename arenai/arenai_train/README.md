# ArenAI training

Full C++ program to train ArenAI agent with LibTorch.

## Installation

Training is only tested on ArchLinux.
Any attempt to build it on other OS will be appreciated.

### ArchLinux

Install dependencies with pacman :
```shell
$ sudo pacman -Sy bullet glm
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
$ cd /path/to/ArenAI/arenai/arenai_train
$ mkdir build && cd build
$ cmake .. && make
```

### Other Linux distro

May work on any kind of decent linux distribution. Many dependencies are resolved with CMake and FetchContent

## Usage

TODO
