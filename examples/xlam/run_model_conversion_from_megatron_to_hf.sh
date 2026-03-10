#!/bin/bash
# Convert Megatron (torch_dist) checkpoint back to HuggingFace safetensors format

BASE_DIR=/fsx/home/jianguozhang

# ============== CONFIGURE YOUR CHECKPOINT ==============
# Original HuggingFace model (for tokenizer, config.json, etc.)
ORIGIN_HF_DIR=${BASE_DIR}/checkpoints/miles/QWEN-3-4B-Instruct-2507

# Megatron checkpoint to convert (iter_XXXXXXX directory)
INPUT_DIR=${BASE_DIR}/checkpoints/miles/xlam-qwen3-4b-instruct-2507--1/iter_0000016

# Output HuggingFace directory
OUTPUT_DIR=${BASE_DIR}/checkpoints/miles/xlam-qwen3-4b-instruct-2507--1-hf

# Optional: Set vocab size if embedding padding needs removal (Qwen3 = 151936)
VOCAB_SIZE=151936

# ============== RUN CONVERSION ==============
echo "Converting Megatron checkpoint to HuggingFace format..."
echo "  Input:  ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Origin HF: ${ORIGIN_HF_DIR}"

cd ${BASE_DIR}/miles

PYTHONPATH=${BASE_DIR}/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir ${INPUT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --origin-hf-dir ${ORIGIN_HF_DIR} \
    --vocab-size ${VOCAB_SIZE} \
    --force

echo ""
echo "Conversion complete! Output saved to: ${OUTPUT_DIR}"
echo ""
echo "You can now load the model with:"
echo "  from transformers import AutoModelForCausalLM"
echo "  model = AutoModelForCausalLM.from_pretrained('${OUTPUT_DIR}')"
