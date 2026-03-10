# xlam: Function Calling RL Training

This example demonstrates SFT and RL training for function calling models with parallel tool support.

## Data Format

Your xlam data should be in JSONL format with OpenAI-compatible messages:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "What's the weather and time in NYC?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"NYC\"}"}},
      {"id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": "{\"city\": \"NYC\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "72F, sunny"},
    {"role": "tool", "tool_call_id": "call_2", "content": "3:45 PM EST"},
    {"role": "assistant", "content": "The weather in NYC is 72F and sunny. The current time is 3:45 PM EST."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {...}}}
  ]
}
```

## Setup

### 1. Prepare Model Checkpoint

```bash
# Download HF checkpoint
huggingface-cli download Qwen/Qwen3-4B-Instruct --local-dir /fsx/home/jianguozhang/models/Qwen3-4B-Instruct

# Convert to Megatron format
cd /fsx/home/jianguozhang/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/fsx/home/jianguozhang/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /fsx/home/jianguozhang/models/Qwen3-4B-Instruct \
    --save /fsx/home/jianguozhang/models/Qwen3-4B-Instruct_torch_dist
```

### 2. Prepare Your Data

Place your xlam training data at:
```
/fsx/home/jianguozhang/data/xlam_train.jsonl
```

### 3. Run SFT

```bash
cd /fsx/home/jianguozhang/slime
export WANDB_KEY=your_wandb_key
bash examples/xlam/xlam_qwen3_sft.sh
```

## Key Configuration

| Parameter | Description |
|-----------|-------------|
| `--prompt-data` | Path to your xlam JSONL data |
| `--input-key messages` | Key for conversation messages |
| `--tool-keys tools` | Key for tool definitions (optional) |
| `--loss-mask-type qwen3` | Tokenizer type for loss masking |
| `--num-epoch 3` | Number of training epochs |
| `--lr 1e-5` | Learning rate |

## Loss Masking

The `MultiTurnLossMaskGenerator` automatically handles:
- Parallel tool_calls in assistant messages
- Multiple tool responses
- Only trains on assistant-generated content (loss_mask=1)
- Skips user/system/tool content (loss_mask=0)

## Parallelism Parameters

### Parameter Overview

| Parameter | What it controls | Memory impact | Throughput impact |
|-----------|-----------------|---------------|-------------------|
| `--tensor-model-parallel-size` (TP) | Model weight distribution across GPUs | ↓ per-GPU model memory | ↓ (communication overhead) |
| `--context-parallel-size` (CP) | Sequence splitting across GPUs | ↓ per-GPU activation memory | ↓ for short seqs, ↑ for long seqs |
| `--max-tokens-per-gpu` | Token budget per GPU per micro-batch | ↑ value = ↑ memory | ↑ value = fewer grad accum steps |
| `--rollout-max-prompt-len` | Max sequence length filter | Filters long samples | More samples if lower |

### GPU Layout Formula

```
Total GPUs = TP × CP × DP

Where:
- TP = tensor-model-parallel-size (model sharding)
- CP = context-parallel-size (sequence sharding)  
- DP = data-parallel-size (computed automatically)
```

**Example with 8 GPUs:**
```
TP=2, CP=1 → DP = 8 / (2×1) = 4 data parallel ranks
TP=2, CP=2 → DP = 8 / (2×2) = 2 data parallel ranks
TP=2, CP=4 → DP = 8 / (2×4) = 1 data parallel rank
```

### Per-GPU Token Calculation

With Context Parallelism, sequences are split across CP GPUs:

| Sample Tokens | CP | Tokens per GPU | Fits budget 16k? |
|--------------|----|-----------------|--------------------|
| 8,000 | 1 | 8,000 | ✓ |
| 8,000 | 2 | 4,000 | ✓ (wasteful overhead) |
| 32,000 | 1 | 32,000 | ❌ exceeds budget |
| 32,000 | 2 | 16,000 | ✓ |
| 64,000 | 2 | 32,000 | ❌ exceeds budget |
| 64,000 | 4 | 16,000 | ✓ |

### Micro-batch Packing

The `max-tokens-per-gpu` controls gradient accumulation via bin-packing:

```python
effective_budget = max_tokens_per_gpu × cp_size

# Samples are packed into micro-batches where:
# sum(sample_tokens) ≤ effective_budget
```

**Example:**
- `max_tokens_per_gpu=16384`, `CP=2` → effective budget = 32k
- A 30k token sample fits in one micro-batch
- After CP split: 15k tokens per GPU (< 16k budget) ✓

### Decision Guide

1. **Filter data** with `--rollout-max-prompt-len` based on your sequence length distribution
2. **Choose CP** so that `max_sequence / CP ≤ max_tokens_per_gpu`
3. **Choose TP** based on model size (larger models need higher TP)
4. **DP is computed**: `DP = total_GPUs / (TP × CP)` — higher DP = faster training
5. **Tune `max-tokens-per-gpu`** for memory: lower = more gradient accumulation, less memory

### Recommended Configurations

| Data Profile | Recommended Settings |
|--------------|---------------------|
| Short sequences (< 8k) | CP=1, higher DP |
| Medium sequences (8k-32k) | CP=1 or CP=2 |
| Long sequences (32k-64k) | CP=2 or CP=4 |
| Mixed lengths | Filter outliers + CP=1, or CP=2 for all |

## Next Steps

After SFT, you can proceed to RL training by creating `xlam_qwen3_rl.sh` based on `retool_qwen3_4b_rl.sh`.
