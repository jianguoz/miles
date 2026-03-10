# export SGLANG_COMMIT="24c91001cf99ba642be791e099d358f4dfe955f5"

# # install sglang
# git clone https://github.com/sgl-project/sglang.git
# cd sglang
# git checkout ${SGLANG_COMMIT}

# export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

# git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
#   cd Megatron-LM/ && git checkout ${MEGATRON_COMMIT} && \
#   pip install -e .


# cd $BASE_DIR/sglang
# git apply $SLIME_DIR/docker/patch/v0.5.7/sglang.patch
# cd $BASE_DIR/Megatron-LM
# git apply $SLIME_DIR/docker/patch/v0.5.7/megatron.patch


export BASE_DIR=${BASE_DIR:-"/fsx/home/jianguozhang"}
# if slime does not exist locally, clone it
if [ ! -d "$BASE_DIR/slime" ]; then
  echo "slime does not exist locally, cloning it"
else
  echo "slime exists locally, skipping cloning"
  export SLIME_DIR=$BASE_DIR/slime
  cd $SLIME_DIR
fi