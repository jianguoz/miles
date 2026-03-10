POD_NAME="node-1"

MAX_CONTEXT_LENGTH=65536

# MODEL_ID="xlam-4b--1-5"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20250702/qwen3_4b_instruct_2507__20250702_xlam_2_filtered_v2__unify_sft_full_training_seq_len_20480_lr_1e-4_bs_2_ga_3_steps_4540_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8014

# MODEL_ID="xlam-32b--7"
# MAX_CONTEXT_LENGTH=32768
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20251018/qwen3_32b__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_5e-6_bs_1_ga_1_steps_2981_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8008

# MODEL_ID="xlam-moe-30b--1"
# MAX_CONTEXT_LENGTH=32768
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20260128/qwen3_30b_a3b_instruct_2507__20260128_xlam_2_filtered_v4__unify_sft_full_training_seq_len_16384_lr_5e-6_bs_1_ga_3_steps_8943_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch_2"
# PORT=8008

# MODEL_ID="qwen3_30b_a3b_instruct_2507"
# MAX_CONTEXT_LENGTH=32768
# ASSIGNED_MODEL_NAME_OR_ID=$MODEL_ID # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/raw/${MODEL_ID}"
# PORT=8009

MODEL_ID="xlam-qwen3-4b-instruct-2507--1-hf"
MAX_CONTEXT_LENGTH=32768
ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}" # e.g., p23-5
MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/miles/${MODEL_ID}"
PORT=8008

# -------------------------------------------------------------
echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
echo "Model path: $MODEL_NAME_OR_PATH"
echo "Port: $PORT"
# echo "Max model length: $MAX_MODEL_LEN"
echo "------------------------------------------------------"


CUDA_VISIBLE_DEVICES=7 nohup vllm serve $MODEL_NAME_OR_PATH \
    --port $PORT \
    --dtype=bfloat16 \
    --tensor-parallel-size=2 \
    --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
    --gpu-memory-utilization=0.90 \
    --enable-auto-tool-choice \
    --max-model-len=$MAX_CONTEXT_LENGTH \
    --tool-call-parser xlam \
    > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &

# MODEL_ID="xlam-4b--6"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20251018/qwen3_4b_instruct_2507__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_1e-5_bs_2_ga_2_steps_4440_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8005

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=5 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &

# MODEL_ID="xlam-4b--6-2"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20251018/qwen3_4b_instruct_2507__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_1e-5_bs_2_ga_1_steps_4440_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8006

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=6 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &


# MODEL_ID="xlam-4b--7"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20251018/qwen3_4b_instruct_2507__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_1e-5_bs_2_ga_2_steps_6201_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8007

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=7 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &


# MODEL_ID="xlam-32b--6"
# MAX_CONTEXT_LENGTH=32768
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen2.5/sft/20251018/qwen2.5_32b_instruct__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_5e-6_bs_1_ga_1_steps_2960_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8009

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=5,6 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --tensor-parallel-size=2 \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &


# MODEL_ID="xlam-32b--5"
# MAX_CONTEXT_LENGTH=32768
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen2.5/sft/20251018/qwen2.5_32b_instruct__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_5e-6_bs_1_ga_1_steps_2884_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8008

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=3,4 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --tensor-parallel-size=2 \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &


# MODEL_ID="xlam-32b--4"
# MAX_CONTEXT_LENGTH=32768
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen2.5/sft/20251018/qwen2.5_32b_instruct__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_5e-6_bs_1_ga_1_steps_3970_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8007

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=5,6 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --tensor-parallel-size=2 \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &

#     # --max-model-len=$MAX_CONTEXT_LENGTH \

# MODEL_ID="xlam-4b--4"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20251018/qwen3_4b_instruct_2507__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_1e-5_bs_2_ga_2_steps_5956_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8006

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=5 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &

#     # --max-model-len=$MAX_CONTEXT_LENGTH \


# MODEL_ID="xlam-4b--4-2"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20251018/qwen3_4b_instruct_2507__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_5e-6_bs_2_ga_2_steps_5956_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8007

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=6 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &


# MODEL_ID="xlam-4b--2-3"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20251018/qwen3_4b_instruct_2507__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_2e-5_bs_2_ga_2_steps_4343_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8004

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=3 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &

#     # --max-model-len=$MAX_CONTEXT_LENGTH \


# MODEL_ID="xlam-4b--3-3"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20251018/qwen3_4b_instruct_2507__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_2e-5_bs_2_ga_2_steps_7850_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8005

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=4 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &


# MODEL_ID="xlam-4b--2-2"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20251018/qwen3_4b_instruct_2507__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_5e-6_bs_2_ga_2_steps_4343_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8002

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=1 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &

#     # --max-model-len=$MAX_CONTEXT_LENGTH \


# MODEL_ID="xlam-4b--3-2"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20251018/qwen3_4b_instruct_2507__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_5e-6_bs_2_ga_2_steps_7850_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8003

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=2 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &



# MODEL_ID="xlam-4b--1-2-ga"
# ASSIGNED_MODEL_NAME_OR_ID="xlam-4b--1-2-ga---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20250702/qwen3_4b_instruct_2507__20250702_xlam_2_filtered_v2__unify_sft_full_training_seq_len_20480_lr_5e-6_bs_2_ga_3_steps_4540_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8002

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=2 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &

#     # --max-model-len=$MAX_CONTEXT_LENGTH \


# MODEL_ID="xlam-4b--1-6"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20250702/qwen3_4b_instruct_2507__20250702_xlam_2_filtered_v3__unify_sft_full_training_seq_len_47104_lr_1e-5_bs_1_ga_3_steps_3649_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8003

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=3 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &


# MODEL_ID="xlam-4b--1-8"
# ASSIGNED_MODEL_NAME_OR_ID="${MODEL_ID}---xlam-3-fc" # e.g., p23-5
# MODEL_NAME_OR_PATH="/fsx/home/jianguozhang/checkpoints/qwen3/sft/20250702/qwen3_4b_instruct_2507__20250702_xlam_2_filtered_v3__unify_sft_full_training_seq_len_47104_lr_2e-5_bs_1_ga_3_steps_3649_pre_bf16_model_id__${MODEL_ID}/final_merged_checkpoint_torch"
# PORT=8001

# # -------------------------------------------------------------
# echo "Serving model ID: $ASSIGNED_MODEL_NAME_OR_ID"
# echo "Model path: $MODEL_NAME_OR_PATH"
# echo "Port: $PORT"
# # echo "Max model length: $MAX_MODEL_LEN"
# echo "------------------------------------------------------"


# CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve $MODEL_NAME_OR_PATH \
#     --port $PORT \
#     --dtype=bfloat16 \
#     --tensor-parallel-size=2 \
#     --served-model-name $ASSIGNED_MODEL_NAME_OR_ID \
#     --gpu-memory-utilization=0.90 \
#     --enable-auto-tool-choice \
#     --max-model-len=$MAX_CONTEXT_LENGTH \
#     --tool-call-parser xlam \
#     > "nohup_files_serve_logs/${ASSIGNED_MODEL_NAME_OR_ID}---${POD_NAME}.nohup" 2>&1 &






