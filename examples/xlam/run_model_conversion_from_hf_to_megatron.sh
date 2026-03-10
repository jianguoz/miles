BASE_DIR=/fsx/home/jianguozhang

# # glm4-9B
# BASE_MODEL_CONVERSION_SCRIPT_NAME="glm4-9B"
# BASE_MODEL_HF_NAME="GLM-Z1-9B-0414"

# # qwen3-30B-A3B
# BASE_MODEL_CONVERSION_SCRIPT_NAME="qwen3-30B-A3B"
# BASE_MODEL_HF_NAME="QWEN-3-30B-A3B"
# BASE_MODEL_HF_NAME="QWEN-3-30B-A3B-Thinking-2507"

# # qwen3-4B-Thinking-2507
# BASE_MODEL_HF_NAME="QWEN-3-4B-Thinking-2507"
# BASE_MODEL_CONVERSION_SCRIPT_NAME="qwen3-4B-Thinking-2507"

# qwen3-4B-Instruct-2507
BASE_MODEL_HF_NAME="QWEN-3-4B-Instruct-2507"
BASE_MODEL_CONVERSION_SCRIPT_NAME="qwen3-4B-Instruct-2507"


echo "Converting ${BASE_MODEL_HF_NAME} from HF to Torch Dist..."


source $BASE_DIR/miles/scripts/models/${BASE_MODEL_CONVERSION_SCRIPT_NAME}.sh
PYTHONPATH=$BASE_DIR/Megatron-LM python $BASE_DIR/miles/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint $BASE_DIR/checkpoints/miles/${BASE_MODEL_HF_NAME} \
    --save $BASE_DIR/checkpoints/miles/${BASE_MODEL_CONVERSION_SCRIPT_NAME}_torch_dist