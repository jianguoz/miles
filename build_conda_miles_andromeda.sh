#!/bin/bash

set -ex

export BASE_DIR=${BASE_DIR:-"/mnt/purple-king/home/jianguo"}

# Use existing micromamba if installed, otherwise install fresh
if ! command -v micromamba &> /dev/null; then
  export MAMBA_ROOT_PREFIX="$BASE_DIR/micromamba"
  yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
fi
export PS1=tmp
mkdir -p ~/.cargo/
touch ~/.cargo/env
source ~/.bashrc

eval "$(micromamba shell hook --shell bash)"
# Only create env if it doesn't exist
if ! micromamba env list | grep -q "miles"; then
  micromamba create -n miles python=3.12 pip -c conda-forge -y
fi
micromamba activate miles

echo "micromamba activated for miles" 
echo "cuda prefix: "$CONDA_PREFIX""
if [ "$CONDA_PREFIX" != "$BASE_DIR/micromamba/envs/miles" ]; then
  echo "Error: CONDA_PREFIX is not set to expected value"
  echo "Expected: $BASE_DIR/micromamba/envs/miles"
  echo "Got: $CONDA_PREFIX"
  exit 1
fi

#----------------------------------------------------------------------------------
# Set CUDA_HOME explicitly to the miles env path
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export SGLANG_COMMIT="24c91001cf99ba642be791e099d358f4dfe955f5"
export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

cd $BASE_DIR

# install cuda 12.9 as it's the default cuda version for torch
micromamba install -n miles cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -c conda-forge -y
micromamba install -n miles -c conda-forge cudnn -y

# prevent installing cuda 13.0 for sglang
pip install cuda-python==13.1.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# install sglang
cd $BASE_DIR
if [ ! -d "$BASE_DIR/sglang" ]; then
  git clone https://github.com/sgl-project/sglang.git
fi
cd sglang
git fetch origin
git checkout ${SGLANG_COMMIT}
pip install -e "python[all]"


pip install cmake ninja

# flash attn
# the newest version megatron supports is v2.7.4.post1
MAX_JOBS=4 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.0
MAX_JOBS=4 NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# megatron
cd $BASE_DIR
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd Megatron-LM
git fetch origin
git checkout ${MEGATRON_COMMIT}
git submodule update --init --recursive
pip install -e .

# install miles and apply patches

# if miles does not exist locally, clone it
if [ ! -d "$BASE_DIR/miles" ]; then
  cd $BASE_DIR
  git clone  https://github.com/radixark/miles.git
  cd miles/
  export miles_DIR=$BASE_DIR/miles
  pip install -e .
else
  export miles_DIR=$BASE_DIR/miles
  cd $miles_DIR
  pip install -e .
fi

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"

# apply patches (reset to clean commit first so patches always apply fresh)
cd $BASE_DIR/sglang
git checkout -- .
git apply $miles_DIR/docker/patch/v0.5.7/sglang.patch

cd $BASE_DIR/Megatron-LM
git checkout -- .
git apply $miles_DIR/docker/patch/v0.5.7/megatron.patch
