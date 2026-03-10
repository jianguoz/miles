# Distributed Training Parallelism Guide

> A comprehensive guide to understanding DP, TP, PP, EP and related concepts in distributed training.

## Table of Contents

1. [Overview: The Four Parallelism Types](#1-overview-the-four-parallelism-types)
2. [Data Parallelism (DP)](#2-data-parallelism-dp)
3. [Tensor Parallelism (TP)](#3-tensor-parallelism-tp)
4. [Pipeline Parallelism (PP)](#4-pipeline-parallelism-pp)
5. [Expert Parallelism (EP)](#5-expert-parallelism-ep)
6. [How They Work Together](#6-how-they-work-together)
7. [Configuration Example](#7-configuration-example)
8. [Trade-off Summary](#8-trade-off-summary)
9. [Batch Size Concepts](#9-batch-size-concepts)
10. [Safeguards for Long Sequences](#10-safeguards-for-long-sequences)
11. [Quick Reference Card](#11-quick-reference-card)

---

## 1. Overview: The Four Parallelism Types

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHAT EACH PARALLELISM SPLITS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DP (Data Parallel)     → Split DATA across GPUs                           │
│                           Each GPU has full model, different data          │
│                                                                             │
│  TP (Tensor Parallel)   → Split LAYER WEIGHTS across GPUs                  │
│                           Each GPU has part of each layer                  │
│                                                                             │
│  PP (Pipeline Parallel) → Split LAYERS across GPUs                         │
│                           Each GPU has different layers                    │
│                                                                             │
│  EP (Expert Parallel)   → Split EXPERTS across GPUs (MoE only)            │
│                           Each GPU has different experts                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Quick Comparison

| Parallelism | What's Split | Communication | Use Case |
|-------------|--------------|---------------|----------|
| **DP** | Data (samples) | AllReduce gradients | Default, most efficient |
| **TP** | Layer weights | AllReduce per layer | Large layers |
| **PP** | Layers (depth) | Point-to-point | Very deep models |
| **EP** | Experts (MoE) | All-to-All | MoE models |

---

## 2. Data Parallelism (DP)

### Concept

Each GPU has a **complete copy** of the model but processes **different data**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA PARALLELISM (DP=4)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Batch of 256 samples split across 4 GPUs:                                 │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   GPU 0     │  │   GPU 1     │  │   GPU 2     │  │   GPU 3     │       │
│  │ Full Model  │  │ Full Model  │  │ Full Model  │  │ Full Model  │       │
│  │ Samples 0-63│  │ Samples     │  │ Samples     │  │ Samples     │       │
│  │             │  │ 64-127      │  │ 128-191     │  │ 192-255     │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                       AllReduce gradients                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How It Works

1. Each GPU loads the **same model weights**
2. Each GPU processes **different data samples**
3. Each GPU computes **local gradients**
4. **AllReduce** averages gradients across all GPUs
5. Each GPU updates weights with averaged gradients

### Trade-offs

| Pros | Cons |
|------|------|
| ✅ Simple to implement | ❌ Each GPU needs full model in memory |
| ✅ Linear speedup with more GPUs | ❌ Communication scales with model size |
| ✅ Good for small-medium models | ❌ Can't train models larger than GPU memory |

### Key Formula

```
Per-GPU batch size = global_batch_size / DP
```

---

## 3. Tensor Parallelism (TP)

### Concept

Split each **layer's weight matrices** across GPUs. All GPUs work on the **same data**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TENSOR PARALLELISM (TP=2)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  One layer's weight matrix W [4096 × 4096] split:                          │
│                                                                             │
│  ┌─────────────────────────────┐                                           │
│  │         W_full              │                                           │
│  └─────────────────────────────┘                                           │
│              ↓ split columns                                               │
│  ┌─────────────────┐    ┌─────────────────┐                                │
│  │     GPU 0       │    │     GPU 1       │                                │
│  │  W[:, :2048]    │    │  W[:, 2048:]    │                                │
│  │  (left half)    │    │  (right half)   │                                │
│  └─────────────────┘    └─────────────────┘                                │
│         │                       │                                          │
│         └───── AllReduce ───────┘  (after EVERY layer!)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How It Works

1. Weight matrices are **split column-wise** across GPUs
2. Each GPU computes partial output for same input
3. **AllReduce** combines partial outputs after each layer
4. This happens for **every transformer layer**

### Trade-offs

| TP Value | Memory/GPU | Communication | DP | When to Use |
|----------|------------|---------------|-----|-------------|
| TP=2 | Higher | Less | Higher | Model fits with TP=2 ✓ |
| TP=4 | Lower | More | Lower | Model doesn't fit with TP=2 |
| TP=8 | Lowest | Most | Lowest | Very large models |

### Key Points

- **Requires fast interconnect** (NVLink preferred, ~600 GB/s)
- **AllReduce after every layer** → high communication overhead
- **Keep TP GPUs on same node** for NVLink speed
- **TP must divide hidden_size evenly**
- PCIe (~32 GB/s) is much slower than NVLink

### Interconnect Comparison

```
NVLink (within node):  ~600 GB/s  ← Use for TP
PCIe 4.0 (across nodes): ~32 GB/s  ← Too slow for TP
```

---

## 4. Pipeline Parallelism (PP)

### Concept

Split model **layers (depth)** across GPUs. Data flows through GPUs sequentially.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE PARALLELISM (PP=4)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  48-layer model split across 4 GPUs:                                       │
│                                                                             │
│  Input                                                                      │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │   GPU 0     │   │   GPU 1     │   │   GPU 2     │   │   GPU 3     │     │
│  │ Layers 0-11 │──▶│ Layers      │──▶│ Layers      │──▶│ Layers      │     │
│  │ (12 layers) │   │ 12-23       │   │ 24-35       │   │ 36-47       │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│                                                                    │        │
│                                                                    ▼        │
│                                                                 Output      │
│                                                                             │
│  Layers per stage = num_layers / PP = 48 / 4 = 12                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Bubble (Idle Time)

```
Time →

GPU 0 (Stage 0): [F0][F1][F2][F3][  ][  ][  ][  ][B3][B2][B1][B0]
GPU 1 (Stage 1): [  ][F0][F1][F2][F3][  ][  ][B3][B2][B1][B0][  ]
GPU 2 (Stage 2): [  ][  ][F0][F1][F2][F3][B3][B2][B1][B0][  ][  ]
GPU 3 (Stage 3): [  ][  ][  ][F0][F1][F2][B2][B1][B0][  ][  ][  ]

F = Forward pass, B = Backward pass
[  ] = Idle time (bubble) = wasted compute!

More PP stages → larger bubble → less efficiency
```

### Trade-offs

| PP Value | Memory/GPU | Communication | Bubble | When to Use |
|----------|------------|---------------|--------|-------------|
| PP=1 | Highest | None | None | Model fits ✓ |
| PP=2 | Half | Low | Small | Model doesn't fit |
| PP=4 | Quarter | Medium | Larger | Very large models |
| PP=8 | 1/8 | Higher | Large | Extreme cases |

### Key Formulas

```
Layers per stage = num_layers / PP
GPUs per model replica = TP × PP
Valid PP values must evenly divide num_layers
```

### Valid PP Values for 48 Layers

PP can be: 1, 2, 3, 4, 6, 8, 12, 16, 24, 48

---

## 5. Expert Parallelism (EP)

### Concept (MoE Models Only)

Split **experts** across GPUs. Tokens are routed to GPUs holding their selected experts.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPERT PARALLELISM (EP=8)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  128 experts split across 8 EP ranks:                                      │
│                                                                             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ...    │
│  │ EP 0   │ │ EP 1   │ │ EP 2   │ │ EP 3   │ │ EP 4   │ │ EP 5   │        │
│  │Experts │ │Experts │ │Experts │ │Experts │ │Experts │ │Experts │        │
│  │ 0-15   │ │ 16-31  │ │ 32-47  │ │ 48-63  │ │ 64-79  │ │ 80-95  │        │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
│                                                                             │
│  Each EP rank holds: 128 / 8 = 16 experts                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Token Routing (All-to-All Communication)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TOKEN ROUTING FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Router selects top-k experts for each token                            │
│     Token "x" → Router → selects experts [3, 17, 45, 67, 89, 102, 115, 127]│
│                                                                             │
│  2. All-to-All: send tokens to GPUs with selected experts                  │
│     Token "x" needs to go to EP ranks 0, 1, 2, 4, 5, 6, 7                  │
│                                                                             │
│  3. Each GPU processes tokens using its local experts                      │
│                                                                             │
│  4. All-to-All: gather results back to original GPUs                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Trade-offs

| EP Value | Experts/GPU | All-to-All Comm | Memory | When to Use |
|----------|-------------|-----------------|--------|-------------|
| EP=4 | 32 | Less | Higher | Smaller cluster |
| EP=8 | 16 | Medium | Medium | Balanced ✓ |
| EP=16 | 8 | More | Lower | Large cluster, tight memory |

### Key Constraint

```
EP × TP ≤ Total GPUs

Example: EP=8, TP=2 → needs at least 8 × 2 = 16 GPUs
```

---

## 6. How They Work Together

### The Master Formula

```
Total GPUs = DP × TP × PP

Therefore: DP = Total GPUs / (TP × PP)
```

### Why This Formula Works

Each "model replica" needs `TP × PP` GPUs working together:
- TP GPUs split each layer horizontally
- PP GPUs split layers vertically

DP tells us how many such replicas fit in our cluster.

### GPU Organization Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              HOW PARALLELISM DIMENSIONS COMBINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Example: 16 GPUs, TP=2, PP=1, EP=8                                        │
│                                                                             │
│  DP = 16 / (2 × 1) = 8                                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ DP/EP Rank 0  │ DP/EP Rank 1  │ ... │ DP/EP Rank 7               │   │
│  ├───────────────┼───────────────┼─────┼─────────────────────────────┤   │
│  │ GPU 0 │ GPU 1 │ GPU 2 │ GPU 3 │ ... │ GPU 14 │ GPU 15            │   │
│  │ TP=0  │ TP=1  │ TP=0  │ TP=1  │     │ TP=0   │ TP=1              │   │
│  │   (tensor)    │   (tensor)    │     │   (tensor)                 │   │
│  │               │               │     │                             │   │
│  │ Experts 0-15  │ Experts 16-31 │     │ Experts 112-127            │   │
│  │ Data batch 0  │ Data batch 1  │     │ Data batch 7               │   │
│  └───────────────┴───────────────┴─────┴─────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Configuration Example

### Example: Qwen3-30B-A3B on 16 GPUs

```bash
# Hardware
--actor-num-nodes 2                    # 2 nodes
--actor-num-gpus-per-node 8            # 8 GPUs per node (total 16)

# Parallelism
--tensor-model-parallel-size 2         # TP=2
--pipeline-model-parallel-size 1       # PP=1
--expert-model-parallel-size 8         # EP=8

# Model
--num-experts 128                      # 128 experts total
--num-layers 48                        # 48 transformer layers

# Training
--global-batch-size 256                # 256 samples per step
```

### Resulting Calculations

| Metric | Calculation | Result |
|--------|-------------|--------|
| Total GPUs | 2 × 8 | 16 |
| Data Parallelism (DP) | 16 / (2 × 1) | 8 |
| Experts per EP rank | 128 / 8 | 16 |
| Layers per PP stage | 48 / 1 | 48 |
| Per-DP batch size | 256 / 8 | 32 samples |

### GPU Layout

```
Node 0 (GPUs 0-7)           Node 1 (GPUs 8-15)
┌───────────────────────┐   ┌───────────────────────┐
│ [0,1] [2,3] [4,5] [6,7]│   │ [8,9] [10,11] [12,13] [14,15] │
│  TP    TP    TP    TP  │   │  TP     TP      TP      TP    │
│  DP0   DP1   DP2   DP3 │   │  DP4    DP5     DP6     DP7   │
│  EP0   EP1   EP2   EP3 │   │  EP4    EP5     EP6     EP7   │
└───────────────────────┘   └───────────────────────┘
```

---

## 8. Trade-off Summary

### Quick Decision Table

| Parameter | ↑ Increase Effect | ↓ Decrease Effect |
|-----------|-------------------|-------------------|
| **TP** | Less memory/GPU, more comm, less DP | More memory/GPU, less comm, more DP |
| **PP** | Less memory/GPU, pipeline bubbles, less DP | More memory/GPU, no bubbles, more DP |
| **EP** | Less memory/GPU, more All-to-All | More memory/GPU, less All-to-All |
| **DP** | More throughput, more total memory | Less throughput, less total memory |

### Decision Flowchart

```
Does model fit with TP=2, PP=1?
│
├── Yes → Use TP=2, PP=1 (maximize DP) ✓
│
└── No
    │
    ├── Try TP=4, PP=1
    │   ├── Fits → Use it
    │   └── No → Continue
    │
    └── Try TP=2, PP=2
        ├── Fits → Use it
        └── No → Increase TP and/or PP further
```

### General Guidelines

1. **Maximize DP** for best throughput
2. **Keep TP GPUs on same node** (NVLink)
3. **Avoid PP if possible** (pipeline bubbles)
4. **EP should match your expert count** and GPU availability

---

## 9. Batch Size Concepts

### Three Different "Batch Sizes"

| Term | Typical Value | Meaning |
|------|---------------|---------|
| `global-batch-size` | 256 | Total samples per training step |
| `rollout-batch-size` | 32 | Prompts per rollout (inference) |
| `n-samples-per-prompt` | 8 | Responses generated per prompt |

### How They Relate

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BATCH SIZE FLOW                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ROLLOUT PHASE (Inference):                                                │
│    rollout-batch-size × n-samples-per-prompt = samples generated           │
│    32 prompts × 8 responses = 256 samples                                  │
│                                                                             │
│  TRAINING PHASE:                                                           │
│    global-batch-size / DP = samples per DP rank                            │
│    256 / 8 = 32 samples per DP rank                                        │
│                                                                             │
│  DYNAMIC BATCHING:                                                         │
│    max-tokens-per-gpu = 28672                                              │
│    Actual samples/GPU = max_tokens / avg_sequence_length                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dynamic Batch Size

With `--use-dynamic-batch-size` and `--max-tokens-per-gpu 28672`:

| Avg Response Length | Samples per GPU |
|---------------------|-----------------|
| 2000 tokens | ~14 samples |
| 4000 tokens | ~7 samples |
| 7000 tokens | ~4 samples |
| 8000 tokens | ~3 samples |

---

## 10. Safeguards for Long Sequences

### Miles Protection Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE LENGTH SAFEGUARDS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 1: Data Loading                                                     │
│  ─────────────────────                                                     │
│  --rollout-max-prompt-len                                                  │
│  → _should_skip_prompt() filters long prompts                              │
│  → Prompts > max_length are SKIPPED                                        │
│  → File: miles/utils/data.py                                               │
│                                                                             │
│  Layer 2: Generation                                                       │
│  ────────────────────                                                      │
│  --rollout-max-response-len (e.g., 8192)                                   │
│  → sglang enforces max_new_tokens                                          │
│  → Long responses are TRUNCATED                                            │
│  → Sample marked as Status.TRUNCATED                                       │
│  → File: miles/rollout/sglang_rollout.py                                   │
│                                                                             │
│  Layer 3: Training                                                         │
│  ────────────────                                                          │
│  --max-tokens-per-gpu (e.g., 28672)                                        │
│  → Dynamic batching adjusts micro-batch size                               │
│  → Prevents OOM during training                                            │
│                                                                             │
│  Layer 4: Model Context Limit                                              │
│  ────────────────────────────                                              │
│  → Model architecture has max context (e.g., 32K-128K)                     │
│  → Positional embeddings break beyond this                                 │
│                                                                             │
│  Layer 5: GPU Memory (Ultimate Safeguard)                                  │
│  ────────────────────────────────────────                                  │
│  → Attention memory ∝ sequence_length²                                     │
│  → OOM crash prevents processing                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Relevant Code Files

| File | Function |
|------|----------|
| `miles/utils/data.py` | `_should_skip_prompt()` - filters long prompts |
| `miles/rollout/sglang_rollout.py` | Enforces `max_new_tokens`, marks truncated |
| `miles/utils/types.py` | `Sample.Status.TRUNCATED` enum |
| `miles/backends/megatron_utils/data.py` | Dynamic batch size handling |

### Recommended Settings

```bash
--rollout-max-prompt-len 4096      # Filter prompts > 4096 tokens
--rollout-max-response-len 8192    # Truncate responses > 8192 tokens
--max-tokens-per-gpu 28672         # Dynamic batching limit
```

---

## 11. Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARALLELISM CHEAT SHEET                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FORMULAS:                                                                 │
│  ─────────                                                                 │
│    Total GPUs = DP × TP × PP                                               │
│    DP = Total GPUs / (TP × PP)                                             │
│    Experts per EP rank = num_experts / EP                                  │
│    Layers per PP stage = num_layers / PP                                   │
│                                                                             │
│  COMMUNICATION TYPES:                                                      │
│  ────────────────────                                                      │
│    DP  → AllReduce gradients (once per step)                              │
│    TP  → AllReduce activations (every layer!)                             │
│    PP  → Point-to-point (pipeline stages)                                 │
│    EP  → All-to-All (token routing)                                       │
│                                                                             │
│  MEMORY vs SPEED TRADE-OFF:                                                │
│  ──────────────────────────                                                │
│    Higher TP/PP → Less memory per GPU, slower training                    │
│    Higher DP    → More memory needed, faster throughput                   │
│                                                                             │
│  BEST PRACTICES:                                                           │
│  ───────────────                                                           │
│    • Maximize DP for throughput                                           │
│    • Keep TP GPUs on same node (NVLink)                                   │
│    • Avoid PP if model fits (pipeline bubbles)                            │
│    • Match EP to your GPU count and memory                                │
│                                                                             │
│  INTERCONNECT SPEEDS:                                                      │
│  ────────────────────                                                      │
│    NVLink (intra-node): ~600 GB/s  ← Use for TP                          │
│    PCIe (inter-node):   ~32 GB/s   ← Too slow for TP                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Glossary

| Term | Definition |
|------|------------|
| **AllReduce** | Collective operation that sums values across GPUs and distributes result to all |
| **All-to-All** | Each GPU sends different data to each other GPU (used in MoE routing) |
| **Bubble** | Idle time in pipeline parallelism when GPUs wait for data |
| **DP** | Data Parallelism - split data across GPUs |
| **EP** | Expert Parallelism - split MoE experts across GPUs |
| **MoE** | Mixture of Experts - model architecture with multiple expert networks |
| **NVLink** | High-speed GPU interconnect (~600 GB/s) |
| **PP** | Pipeline Parallelism - split model layers across GPUs |
| **TP** | Tensor Parallelism - split layer weights across GPUs |
| **top-k** | Number of experts selected per token in MoE |

---

*Document created from miles training discussions. Last updated: January 2026.*



