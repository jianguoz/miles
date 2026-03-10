#!/bin/bash
# xlam SFT training script
# Adapted from retool example for function calling with parallel tool support

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

BASE_DIR=/fsx/home/jianguozhang
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_DIR="${BASE_DIR}/miles"

# Load W&B API key from config file
if [ -f "$BASE_DIR/jianguozhang/notes/access/wandb.json" ]; then
   WANDB_KEY=$(python3 -c "import json; print(json.load(open('$BASE_DIR/jianguozhang/notes/access/wandb.json'))['api_key'])")
fi

# Source model config - change this for different models
source "${MILES_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

# ============== MODIFY THESE PATHS FOR YOUR SETUP ==============
# PyTorch distributed checkpoint (torch_dist) supports automatic resharding,
# so existing checkpoint works with different TP/CP settings.
HF_CHECKPOINT=${BASE_DIR}/checkpoints/miles/QWEN-3-4B-Instruct-2507/
MEGATRON_CHECKPOINT=${BASE_DIR}/checkpoints/miles/QWEN-3-4B-Instruct-2507_torch_dist
SAVE_DIR=${BASE_DIR}/checkpoints/miles/xlam-qwen3-4b-instruct-2507--1/


# Validate checkpoints exist
if [ ! -d "$HF_CHECKPOINT" ]; then
   echo "ERROR: HuggingFace checkpoint not found: $HF_CHECKPOINT"
   exit 1
fi
if [ ! -d "$MEGATRON_CHECKPOINT" ]; then
   echo "ERROR: Megatron checkpoint not found: $MEGATRON_CHECKPOINT"
   echo "Please run model conversion first: bash run_model_conversion.sh"
   exit 1
fi
echo "Checkpoints paths validated successfully."

CKPT_ARGS=(
   # HuggingFace checkpoint (for tokenizer)
   --hf-checkpoint ${HF_CHECKPOINT}
   # Megatron checkpoint (converted from HF)
   --ref-load ${MEGATRON_CHECKPOINT}
   # Output directory for SFT checkpoint
   --save ${SAVE_DIR}
   --save-interval 20
)

# ============== XLAM DATA CONFIG ==============
DATA_FILE=${SCRIPT_DIR}/gorilla_multi__fit_len_43008.jsonl
if [ ! -f "$DATA_FILE" ]; then
   echo "ERROR: Training data not found: $DATA_FILE"
   exit 1
fi
echo "🎉 Training data path validated: $DATA_FILE"

SFT_ARGS=(
   --rollout-function-path miles.rollout.sft_rollout.generate_rollout
   # Multi-turn function call data (gorilla format)
   --prompt-data ${DATA_FILE}
   # Key in your data for messages array
   --input-key messages
   # Key in your data for tools array
   --tool-key tools
   # Loss mask type for Qwen3 tokenizer (supports parallel tool_calls)
   --loss-mask-type qwen3
   # Max sequence length filter (samples >43k will be filtered)
   --rollout-max-prompt-len 43008
   --rollout-shuffle
   --num-epoch 1
   # Rollout batch must be >= global batch size
   # Gradient accumulation is controlled by max-tokens-per-gpu below
   --rollout-batch-size 32
   --global-batch-size 32

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   # 4B model is small, fits on single GPU. Use CP=2 to split sequences.
   # TP=1, CP=2 = 2 GPUs per sample, DP = 8/2 = 4
   # 43k tokens / CP=2 = 21.5k per GPU (safe margin)
   --tensor-model-parallel-size 1
   # --sequence-parallel  # Not needed with TP=1
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # Token budget per GPU: 4B model should handle 43k tokens fine
   # Reduce if OOM (e.g., 32768 or 16384)
   --max-tokens-per-gpu 43008
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project xlam-training
   --wandb-group xlam-qwen3-4b-instruct-2507--1
   --disable-wandb-random-suffix
   --wandb-key ${WANDB_KEY}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# launch the master node of ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"conda\": \"$BASE_DIR/micromamba/envs/miles\",
  \"env_vars\": {
    \"PYTHONPATH\": \"$BASE_DIR/miles:$BASE_DIR/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

cd ${MILES_DIR}

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${MILES_DIR}/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}
