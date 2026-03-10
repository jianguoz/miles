#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

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

# Load W&B API key from config file
if [ -f "$BASE_DIR/jianguozhang/notes/access/wandb.json" ]; then
   WANDB_KEY=$(python3 -c "import json; print(json.load(open('$BASE_DIR/jianguozhang/notes/access/wandb.json'))['api_key'])")
fi

source "$BASE_DIR/miles/scripts/models/glm4-9B.sh"

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/checkpoints/miles/GLM-Z1-9B-0414/
   --ref-load $BASE_DIR/checkpoints/miles/GLM-Z1-9B-0414_torch_dist
   --load $BASE_DIR/checkpoints/miles/GLM-Z1-9B-0414_miles/
   --save $BASE_DIR/checkpoints/miles/GLM-Z1-9B-0414_miles/
   --save-interval 40
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/checkpoints/miles/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

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

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
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
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-training
   --wandb-group glm4-9B-jianguo-$(date +%Y%m%d-%H%M%S)
   --disable-wandb-random-suffix
#    --wandb-run-name glm4-9B-run-$(date +%Y%m%d-%H%M%S)
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
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
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
# symlink: ln -s /fsx/home/jianguozhang/micromamba/envs/miles/lib /fsx/home/jianguozhang/micromamba/envs/miles/lib64
RUNTIME_ENV_JSON="{
  \"conda\": \"$BASE_DIR/micromamba/envs/miles\",
  \"env_vars\": {
    \"PYTHONPATH\": \"$BASE_DIR/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# \"LIBRARY_PATH\": \"$BASE_DIR/micromamba/envs/miles/lib:$BASE_DIR/micromamba/envs/miles/lib/stubs:$BASE_DIR/micromamba/envs/miles/targets/x86_64-linux/lib\",
# \"LD_LIBRARY_PATH\": \"$BASE_DIR/micromamba/envs/miles/lib:$BASE_DIR/micromamba/envs/miles/lib/stubs:$BASE_DIR/micromamba/envs/miles/targets/x86_64-linux/lib\"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 $BASE_DIR/miles/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
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
