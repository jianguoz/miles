from transformers import AutoTokenizer, AutoConfig
# from actionstudio.src.foundation_modeling.utils.common import *

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you, what is the current time?"},
    {
        "role": "assistant",
        "content": "Sure, I can help you to <think>YANMEI\nHELLO</think> check it.",
        "reasoning_content": "JIANGUO ZHANG\nHELLO",
        # "reasoning_content": "I will </think>JIANGUO ZHANG\n</think> and the available stocks.",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "arguments": "{}"
                },
                "id": "633159740"
            },
            {
                "type": "function",
                "function": {
                    "name": "get_available_stocks",
                    "arguments": "{\"sector\": \"Technology\"}"
                },
                "id": "294116824"
            }
        ]
    },
    {
        "role": "tool",
        "content": "{\"current_time\": \"10:30 AM\"}",
        "tool_call_id": "633159740"
    },
    {
        "role": "tool",
        "content": "{\"stock_list\": [\"AAPL\", \"GOOG\", \"MSFT\", \"NVDA\", \"AAPL\", \"GOOG\", \"MSFT\", \"NVDA\", \"LGY\", \"HW\", \"OLMK\", \"O\", \"ZTMG\", \"BZHAI\", \"CEXPE\", \"MQZRF\", \"XBKC\", \"W\", \"RHWL\", \"CL\", \"KO\", \"WG\", \"ST\", \"RNUUI\", \"NFLP\", \"LBCV\", \"U\", \"Z\", \"ABBBQ\", \"QCZHX\", \"TMMOX\"}",
        "tool_call_id": "294116824"
    },
    {"role": "assistant", "content": "The current time is 2026-01-29 10:00:00"},
    {"role": "user", "content": "What is the capital of the moon?"},
    {"role":"assistant","content":"hello<think>\ngot it think\n</think>\n\nDone!"},

    # {
    #     "role": "assistant", 
    #     "content": "The capital of the moon is <think>YANMEI\nHELLO</think> the moon!",
    #     "reasoning_content": "JIANGUO ZHANG\nHELLO",
    #     "tool_calls": [
    #         {
    #             "type": "function",
    #             "function": {"name": "get_current_time", "arguments": "{}"}
    #         }
    #     ]
    # },
]
messages = [
    # {"role": "user", "content": "<|im_start|>user\nWhat is the weather in New York?<|im_start|>assistant\n"},
    {"role": "user", "content": "What is the weather in New York?"},
    {
        "role": 'assistant',
        "content": "<think>\nThe user is asking about the weather in New York. I should use the weather tool to get this information.\n</think>\nI'll check the current weather in New York for you.",
        "tool_calls": [
            {
                "function": {
                    "name": 'get_weather',
                    "arguments": {
                        "location": 'New York',
                        "unit": 'celsius'
                    }
                }
            },
            {
                "function": {
                    "name": 'get_weather',
                    "arguments": {
                        "location": 'Beijing',
                        "unit": 'celsius'
                    }
                }
            }
        ]
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "arguments": "{}"},
        "id": "633159740"
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_stocks",
            "arguments": "{\"sector\": \"Technology\"}"
        },
        "id": "294116824"
    }
]

# tokenizer = AutoTokenizer.from_pretrained("/fsx/home/jianguozhang/checkpoints/qwen3/sft/20260128/qwen3_30b_a3b_instruct_2507__20260128_xlam_2_filtered_v4__unify_sft_full_training_seq_len_16384_lr_5e-6_bs_1_ga_3_steps_8943_pre_bf16_model_id__xlam-moe-30b--1/final_merged_checkpoint_torch/")
tokenizer = AutoTokenizer.from_pretrained("/fsx/home/jianguozhang/checkpoints/qwen3/raw/qwen3_4b_thinking_2507")
# tokenizer = AutoTokenizer.from_pretrained("/fsx/home/jianguozhang/checkpoints/qwen3/sft/20251018/qwen3_4b_instruct_2507__20251018_xlam_2_filtered_v4__unify_sft_full_training_seq_len_32768_lr_5e-6_bs_2_ga_3_steps_4343_pre_bf16_model_id__xlam-4b--2/final_merged_checkpoint_torch/")
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=True)

prompt = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    # add_generation_prompt=True
    )

print(prompt)

"""
Summary for qwen3_4b_❤️thinking❤️_2507:
+---------------------------+-------------------+----------------------------------------------------------+
| Condition                 | Message Position  | Behavior                                                 |
+---------------------------+-------------------+----------------------------------------------------------+
| WITH reasoning_content    | non-last assistant| reasoning_content discarded, content shown as-is         |
|                           | last assistant    | reasoning_content wrapped in <think></think>, then content shown as-is |
+---------------------------+-------------------+----------------------------------------------------------+
| WITHOUT reasoning_content | non-last assistant| everything up to </think> discarded (incl. text before   |
| (when <think> in content) |                   | <think>), only text after </think> shown                 |
|                           | last assistant    | thinking EXTRACTED into <think> tags, text after </think>|
|                           |                   | shown, text before <think> discarded                     |
+---------------------------+-------------------+----------------------------------------------------------+


========With reasoning content========
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_current_time", "arguments": "{}"}, "id": "633159740"}
{"type": "function", "function": {"name": "get_available_stocks", "arguments": "{\"sector\": \"Technology\"}"}, "id": "294116824"}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Hello, how are you, what is the current time?<|im_end|>
<|im_start|>assistant
Sure, I can help you to <think>YANMEI
HELLO</think> check it.
<tool_call>
{"name": "get_current_time", "arguments": {}}
</tool_call>
<tool_call>
{"name": "get_available_stocks", "arguments": {"sector": "Technology"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"current_time": "10:30 AM"}
</tool_response>
<tool_response>
{"stock_list": ["AAPL", "GOOG", "MSFT", "NVDA", "AAPL", "GOOG", "MSFT", "NVDA", "LGY", "HW", "OLMK", "O", "ZTMG", "BZHAI", "CEXPE", "MQZRF", "XBKC", "W", "RHWL", "CL", "KO", "WG", "ST", "RNUUI", "NFLP", "LBCV", "U", "Z", "ABBBQ", "QCZHX", "TMMOX"}
</tool_response><|im_end|>
<|im_start|>assistant
The current time is 2026-01-29 10:00:00<|im_end|>
<|im_start|>user
What is the capital of the moon?<|im_end|>
<|im_start|>assistant
<think>
JIANGUO ZHANG
HELLO
</think>

The capital of the moon is <think>YANMEI
HELLO</think> the moon!
<tool_call>
{"name": "get_current_time", "arguments": {}}
</tool_call><|im_end|>
<|im_start|>assistant
<think>


========Without reasoning content========
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_current_time", "arguments": "{}"}, "id": "633159740"}
{"type": "function", "function": {"name": "get_available_stocks", "arguments": "{\"sector\": \"Technology\"}"}, "id": "294116824"}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Hello, how are you, what is the current time?<|im_end|>
<|im_start|>assistant
 check it.
<tool_call>
{"name": "get_current_time", "arguments": {}}
</tool_call>
<tool_call>
{"name": "get_available_stocks", "arguments": {"sector": "Technology"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"current_time": "10:30 AM"}
</tool_response>
<tool_response>
{"stock_list": ["AAPL", "GOOG", "MSFT", "NVDA", "AAPL", "GOOG", "MSFT", "NVDA", "LGY", "HW", "OLMK", "O", "ZTMG", "BZHAI", "CEXPE", "MQZRF", "XBKC", "W", "RHWL", "CL", "KO", "WG", "ST", "RNUUI", "NFLP", "LBCV", "U", "Z", "ABBBQ", "QCZHX", "TMMOX"}
</tool_response><|im_end|>
<|im_start|>assistant
The current time is 2026-01-29 10:00:00<|im_end|>
<|im_start|>user
What is the capital of the moon?<|im_end|>
<|im_start|>assistant
<think>
YANMEI
HELLO
</think>

 the moon!
<tool_call>
{"name": "get_current_time", "arguments": {}}
</tool_call><|im_end|>
<|im_start|>assistant
<think>

"""