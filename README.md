# ArenAI

A battle-royal game with trained agent which controls tanks in realistic physic world.

## Description

Each tank receives the rendered frame of its camera as input, and the agent is trained to fire and hit enemies.

When the agent is trained (PPO or PPG), you can fight it through the tanks it handles.

## Documentation

| | |
|---|---|
| [docs/BUILD.md](docs/BUILD.md) | dependencies, build (Linux, Windows, macOS), tests |
| [docs/USAGE.md](docs/USAGE.md) | play, train an agent, evaluate a checkpoint |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | modules, class diagrams, frame sequences |

## Quick start

Linux, dependencies and LibTorch installed ([docs/BUILD.md](docs/BUILD.md)) :
```bash
git clone https://github.com/Ipsedo/ArenAI.git
cd ArenAI
./build_linux.sh
./build/arenai_desktop/arenai_desktop
```

Windows (Visual Studio 2022 with C++ workload), in PowerShell :
```powershell
.\install_dependencies.ps1
.\build_windows.ps1
```

Train an agent ([docs/USAGE.md](docs/USAGE.md)) :
```bash
./build/arenai_agent/arenai_agent_train --output_folder outputs/train_001_ppo --resources_folder resources --cuda ppo
```

### macOS

There is no local build for macOS yet : use the `arenai-<version>_macos-arm64.zip` archive of a [release](https://github.com/Ipsedo/ArenAI/releases) (Apple Silicon, experimental).

The application is not signed with an Apple developer account, so macOS blocks it once downloaded.
Unzip the archive, then open a Terminal in the folder containing `ArenAI.app` and remove the quarantine flag :
```bash
xattr -dr com.apple.quarantine ArenAI.app
```

You can now launch `ArenAI.app` with a double click.

## Known issues

Any bug, build failed, etc. reports will be really appreciated.
Create your issue if you want to participate !

### GNOME system monitor extension

The `system-monitor` extension is causing freeze periodically. The fix is to disable it when playing :
```bash
# on ArchLinux
gnome-extensions disable system-monitor@gnome-shell-extensions.gcampax.github.com
```
