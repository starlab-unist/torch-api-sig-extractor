# Torch API Signature Extractor

Torch API Signature Extractor is a tool designed to extract API signatures (i.e., type information) from the [PyTorch](https://github.com/pytorch/pytorch) source code. The extracted signatures serve as input for the [PathFinder Driver Generator](https://github.com/starlab-unist/pathfinder-driver-generator), which automatically generates test drivers for [PathFinder](https://github.com/starlab-unist/pathfinder).

## Setup

```bash
apt-get update && apt-get install -y cmake ninja-build

git clone --depth 1 -b extractor git@github.com:starlab-unist/torch-api-sig-extractor.git
cd torch-api-sig-extractor && mkdir build && cd build
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" -DCMAKE_INSTALL_PREFIX=. -DCMAKE_BUILD_TYPE=Debug ../llvm
ninja
```

## Usage

The extractor assumes the presence of a PyTorch directory built from the source code. The following command will generate `torch-api-signatures.json` file.

```bash
build/bin/extractor -p $PYTORCH_HOME/build/ $PYTORCH_HOME/torch/csrc/api/include/torch/torch.h
```
