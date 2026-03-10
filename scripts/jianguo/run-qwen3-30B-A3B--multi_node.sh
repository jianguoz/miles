#!/bin/bash
#
# Qwen3-30B-A3B Multi-Node Training Script
#
# Supports 1-N nodes with auto-detection from hostfile.
# Single node: TP=4, EP=8, colocate mode
# Multi-node:  TP=2, EP=8, separate rollout GPUs
#
# 8 GPUs per engine = 1 engine per node (no cross-node communication)
#
# TESTED: recompute-num-layers=0 caused RAM OOM (worker hit 1.6TB+)
# KEEP: recompute-num-layers=1, max-tokens=28672 (safe + faster)
#

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

BASE_DIR=/fsx/home/jianguozhang

# Hostfile and master node configuration
HOSTFILE="$BASE_DIR/miles/scripts/jianguo/hostfiles/qwen3-30B-A3B"
MASTER_ADDR="10.3.76.98"

# Verify hostfile exists
if [ ! -f "$HOSTFILE" ]; then
   echo "ERROR: Hostfile not found: $HOSTFILE"
   exit 1
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Load W&B API key from config file
if [ -f "$BASE_DIR/jianguozhang/notes/access/wandb.json" ]; then
   WANDB_KEY=$(python3 -c "import json; print(json.load(open('$BASE_DIR/jianguozhang/notes/access/wandb.json'))['api_key'])")
fi
source "$BASE_DIR/miles/scripts/models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/checkpoints/miles/QWEN-3-30B-A3B/
   --ref-load $BASE_DIR/checkpoints/miles/QWEN-3-30B-A3B_torch_dist
   --load $BASE_DIR/checkpoints/miles/QWEN-3-30B-A3B_miles/
   --save $BASE_DIR/checkpoints/miles/QWEN-3-30B-A3B_miles/
   --save-interval 60
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/checkpoints/miles/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   # Keep rollout-batch-size=32 to avoid system RAM OOM
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   # global-batch-size = rollout-batch-size × n-samples-per-prompt = 32 × 8 = 256
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime $BASE_DIR/checkpoints/miles/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-training
   # Updated group name for optimized 2-node run
   --wandb-group QWEN-3-30B-A3B-2-nodes-optimized-$(date +%Y%m%d-%H%M%S)
   --disable-wandb-random-suffix
   --wandb-key ${WANDB_KEY}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Start Ray workers on other nodes via SSH
# NOTE: Requires passwordless SSH. Run: cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
for WORKER_IP in $(awk '{print $1}' "$HOSTFILE"); do
  # Skip master node (already started above) and empty lines
  if [[ "$WORKER_IP" == "$MASTER_ADDR" ]] || [[ -z "$WORKER_IP" ]]; then
    continue
  fi
  echo "Starting Ray worker on ${WORKER_IP}"
  ssh -o StrictHostKeyChecking=no ${WORKER_IP} \
    "pkill -9 sglang 2>/dev/null; ray stop --force 2>/dev/null; pkill -9 python 2>/dev/null; export PATH=${BASE_DIR}/.local/bin:\$PATH; export MAMBA_ROOT_PREFIX=${BASE_DIR}/micromamba; eval \"\$(${BASE_DIR}/.local/bin/micromamba shell hook --shell bash)\"; micromamba activate miles; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats" &
done
wait

# Wait for workers to join the Ray cluster
echo "Waiting for workers to join Ray cluster..."
sleep 10

# Build the runtime environment JSON with proper variable substitution
# Note: Using env_vars instead of conda to avoid 'conda not found' error on micromamba systems
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PATH\": \"$BASE_DIR/micromamba/envs/miles/bin:$BASE_DIR/.local/bin:/usr/local/bin:/usr/bin:/bin\",
    \"LD_LIBRARY_PATH\": \"$BASE_DIR/micromamba/envs/miles/lib:$BASE_DIR/micromamba/envs/miles/lib64:/usr/local/cuda/lib64\",
    \"PYTHONPATH\": \"$BASE_DIR/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"CONDA_PREFIX\": \"$BASE_DIR/micromamba/envs/miles\",
    \"VIRTUAL_ENV\": \"$BASE_DIR/micromamba/envs/miles\"
  }
}"

# Auto-detect number of nodes: master + workers not equal to master
WORKER_COUNT=$(awk -v master="$MASTER_ADDR" '$1 != master && $1 != "" {count++} END {print count+0}' "$HOSTFILE")
NUM_NODES=$((1 + WORKER_COUNT))  # 1 master + workers
GPUS_PER_NODE=8
TOTAL_ROLLOUT_GPUS=$((NUM_NODES * GPUS_PER_NODE))
echo "❤️ Detected ❤️ ${NUM_NODES} nodes (1 master + ${WORKER_COUNT} workers), total ${TOTAL_ROLLOUT_GPUS} GPUs"

# Set TP based on number of nodes, EP stays at 8 (like 235B uses EP=16 for any node count)
if [ "$NUM_NODES" -eq 1 ]; then
   TP_SIZE=4  # Single node: more TP for memory
else
   TP_SIZE=2  # Multi-node: less TP, more DP
fi
EP_SIZE=8  # Fixed: 128 experts / 8 = 16 experts per GPU (fits easily for 30B)
echo "Using TP=${TP_SIZE}, EP=${EP_SIZE}"

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1    # KEEP: recompute=0 caused RAM OOM on worker

   --use-dynamic-batch-size
   --max-tokens-per-gpu 28672  # OPTIMIZED: +40% from 20480 (using GPU headroom)
)

# Set SGLANG_ARGS based on detected GPU count
SGLANG_ARGS=(
   # 8 GPUs per engine = 1 engine per node (no cross-node communication)
   --rollout-num-gpus-per-engine 8
   # OPTIMIZED: 0.75 (was 0.7) - slightly more KV cache with H200 memory
   --sglang-mem-fraction-static 0.75
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # Note: DeepEP removed - not installed on this machine
)

# Use colocate to share GPUs between training and rollout
# (separate rollout GPUs would require 2x the GPUs: 16 training + 16 rollout = 32)
MULTI_NODE_ARGS=(--colocate)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 $BASE_DIR/miles/train.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${GPUS_PER_NODE} \
   ${MULTI_NODE_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

