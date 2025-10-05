# PhyVR

An Android game with AI

## Requirements

### Android game

You must have `python3.12` in your `PATH` and have CMake version equal to `3.30.3`.

ExecuTorch requires a specific version of buck2 :
```shell
cargo +nightly-2025-02-16 install --git https://github.com/facebook/buck2.git --tag 2025-05-06 buck2
```

Then export cargo binaries to `PATH` (like in your `.bashrc`) :
```shell
# inside .bashrc
export PATH="$PATH:$HOME/.cargo/bin"
```

You are now ready to build with gradle !

### Training agent

TODO
