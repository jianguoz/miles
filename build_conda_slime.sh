#!/bin/bash

set -ex

export BASE_DIR=${BASE_DIR:-"/fsx/home/jianguozhang"}

# # Use existing micromamba if installed, otherwise install fresh
# if ! command -v micromamba &> /dev/null; then
#   export MAMBA_ROOT_PREFIX="$BASE_DIR/micromamba"
#   yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# fi
# export PS1=tmp
# mkdir -p ~/.cargo/
# touch ~/.cargo/env
# source ~/.bashrc

# eval "$(micromamba shell hook --shell bash)"
# micromamba create -n slime python=3.12 pip -c conda-forge -y
# micromamba activate slime

# echo "micromamba activated for slime" 

#----------------------------------------------------------------------------------
# Set CUDA_HOME explicitly to the slime env path
export CUDA_HOME="/fsx/home/jianguozhang/micromamba/envs/slime"
export SGLANG_COMMIT="24c91001cf99ba642be791e099d358f4dfe955f5"
export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

cd $BASE_DIR

# install cuda 12.9 as it's the default cuda version for torch
micromamba install -n slime cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
micromamba install -n slime -c conda-forge cudnn -y

# prevent installing cuda 13.0 for sglang
pip install cuda-python==13.1.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# install sglang
mkdir -p $BASE_DIR/dependencies
cd $BASE_DIR/dependencies/
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout ${SGLANG_COMMIT}
# Install the python packages
pip install -e "python[all]"


pip install cmake ninja

# flash attn
# the newest version megatron supports is v2.7.4.post1
CUDA_HOME=/fsx/home/jianguozhang/micromamba/envs/slime MAX_JOBS=4 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
CUDA_HOME=/fsx/home/jianguozhang/micromamba/envs/slime pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.0
CUDA_HOME=/fsx/home/jianguozhang/micromamba/envs/slime MAX_JOBS=4 NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# megatron
cd $BASE_DIR/dependencies/
git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
  cd Megatron-LM/ && git checkout ${MEGATRON_COMMIT} && \
  pip install -e .

# install slime and apply patches

# if slime does not exist locally, clone it
if [ ! -d "$BASE_DIR/slime" ]; then
  cd $BASE_DIR
  git clone  https://github.com/THUDM/slime.git
  cd slime/
  export SLIME_DIR=$BASE_DIR/slime
  pip install -e .
else
  export SLIME_DIR=$BASE_DIR/slime
  cd $SLIME_DIR
  pip install -e .
fi

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"

# apply patch
cd $BASE_DIR/dependencies/sglang
git apply $SLIME_DIR/docker/patch/v0.5.7/sglang.patch
cd $BASE_DIR/dependencies/Megatron-LM
git apply $SLIME_DIR/docker/patch/v0.5.7/megatron.patch
