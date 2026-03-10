BASE_DIR=/fsx/home/jianguozhang

# # glm4-9B
# BASE_SOURCE_MODEL_BASH_SCRIPT_NAME="glm4-9B"
# BASE_SOURCE_MODEL_HF_NAME="GLM-Z1-9B-0414"

# qwen3-30B-A3B
BASE_SOURCE_MODEL_BASH_SCRIPT_NAME="qwen3-30B-A3B"
# BASE_SOURCE_MODEL_HF_NAME="QWEN-3-30B-A3B"
BASE_SOURCE_MODEL_HF_NAME="QWEN-3-30B-A3B-Thinking-2507"

echo "Converting ${BASE_SOURCE_MODEL_HF_NAME} from HF to Torch Dist..."

BASE_SOURCE_MODEL_TORCH_DIST_NAME="${BASE_SOURCE_MODEL_HF_NAME}_torch_dist"
source $BASE_DIR/miles/scripts/models/${BASE_SOURCE_MODEL_BASH_SCRIPT_NAME}.sh
PYTHONPATH=$BASE_DIR/Megatron-LM python $BASE_DIR/miles/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint $BASE_DIR/checkpoints/miles/${BASE_SOURCE_MODEL_HF_NAME} \
    --save $BASE_DIR/checkpoints/miles/${BASE_SOURCE_MODEL_TORCH_DIST_NAME}