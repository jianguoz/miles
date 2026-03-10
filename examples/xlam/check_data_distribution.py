import json
from transformers import AutoTokenizer
from collections import defaultdict

MAX_LENGTH = 43008 #49152
INPUT_FILE = "gorilla_multi.jsonl"
OUTPUT_FILE = f"gorilla_multi__fit_len_{MAX_LENGTH}.jsonl"

tokenizer = AutoTokenizer.from_pretrained("/fsx/home/jianguozhang/checkpoints/qwen3/raw/qwen3_4b_instruct_2507")

# Read all samples and compute lengths
samples = []
lengths = []
with open(INPUT_FILE) as f:
    for line in f:
        sample = json.loads(line)
        # apply_chat_template with tokenize=True returns list of token IDs directly for transformers < 5.0.0
        token_ids = tokenizer.apply_chat_template(
            sample["messages"], 
            tools=sample.get("tools"), 
            add_generation_prompt=True, 
            tokenize=True
        )
        length = len(token_ids)
        samples.append(sample)
        lengths.append(length)

print(f"Original data: {len(samples)} samples")
print(f"Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.0f}")

# Count samples by length bucket
count_lengths = defaultdict(int)
for length in lengths:
    if length > MAX_LENGTH:
        count_lengths[f">{MAX_LENGTH}"] += 1
print(f"Samples > {MAX_LENGTH}: {count_lengths}")

# Filter out long samples
filtered_samples = []
filtered_lengths = []
removed_count = 0
for sample, length in zip(samples, lengths):
    if length <= MAX_LENGTH:
        filtered_samples.append(sample)
        filtered_lengths.append(length)
    else:
        print(sample["id"], length)
        removed_count += 1

print(f"\nFiltered data: {len(filtered_samples)} samples (removed {removed_count})")
print(f"Min: {min(filtered_lengths)}, Max: {max(filtered_lengths)}, Mean: {sum(filtered_lengths)/len(filtered_lengths):.0f}")

# Save filtered data
with open(OUTPUT_FILE, "w") as f:
    for sample in filtered_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"\nSaved filtered data to: {OUTPUT_FILE}")

"""
Min: 3450, Max: 71795, Mean: 7927
defaultdict(<class 'int'>, {'>34816': 36})
defaultdict(<class 'int'>, {'>=32768': 36})

gorilla_multi----multi_turn_long_context_100 34938
gorilla_multi----multi_turn_long_context_101 36295
gorilla_multi----multi_turn_long_context_105 39021
gorilla_multi----multi_turn_long_context_107 37091
gorilla_multi----multi_turn_long_context_109 43136
gorilla_multi----multi_turn_long_context_110 42709
gorilla_multi----multi_turn_long_context_112 43357
gorilla_multi----multi_turn_long_context_113 71798
gorilla_multi----multi_turn_long_context_114 40268
gorilla_multi----multi_turn_long_context_115 36754
gorilla_multi----multi_turn_long_context_116 43368
gorilla_multi----multi_turn_long_context_118 40923
gorilla_multi----multi_turn_long_context_119 40296
gorilla_multi----multi_turn_long_context_120 36989
gorilla_multi----multi_turn_long_context_123 36963
gorilla_multi----multi_turn_long_context_124 39270
gorilla_multi----multi_turn_long_context_126 40850
gorilla_multi----multi_turn_long_context_127 41084
gorilla_multi----multi_turn_long_context_128 40921
gorilla_multi----multi_turn_long_context_129 38339
gorilla_multi----multi_turn_long_context_132 36754
gorilla_multi----multi_turn_long_context_133 40098
gorilla_multi----multi_turn_long_context_136 39054
gorilla_multi----multi_turn_long_context_137 40296
gorilla_multi----multi_turn_long_context_142 39226
gorilla_multi----multi_turn_long_context_145 38095
gorilla_multi----multi_turn_long_context_146 40761
gorilla_multi----multi_turn_long_context_148 38393
gorilla_multi----multi_turn_long_context_103 42472
gorilla_multi----multi_turn_long_context_108 42928
gorilla_multi----multi_turn_long_context_134 38945
gorilla_multi----multi_turn_long_context_135 44897
gorilla_multi----multi_turn_long_context_147 36877
gorilla_multi----multi_turn_long_context_121 41197
gorilla_multi----multi_turn_long_context_138 38249
gorilla_multi----multi_turn_long_context_131 42400

Samples >= 36864: defaultdict(<class 'int'>, {'>=36864': 32})
Samples >= 43008: defaultdict(<class 'int'>, {'>=43008': 5})

Samples > 43008: defaultdict(<class 'int'>, {'>43008': 5})
Filtered data: 574 samples (removed 5)
Min: 3453, Max: 42928, Mean: 7569
"""
