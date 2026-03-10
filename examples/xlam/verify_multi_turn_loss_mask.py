from transformers import AutoTokenizer

class MultiTurnLossMaskGenerator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.system_message_length, self.gen_token_length = self.get_system_message_length()
        print(f"system_message_length={self.system_message_length}, gen_token_length={self.gen_token_length}")

    def find_all_sublist_indices(self, main_list, sublist):
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def get_system_message_length(self) -> tuple[int, int]:
        """Calculate the length of system message prefix and generation prompt tokens."""
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        
        # gen_token_length = tokens added by add_generation_prompt (e.g., "<|im_start|>assistant\n")
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages, add_special_tokens=False, tokenize=True, add_generation_prompt=True
            )
        ) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def gen_multi_turn_loss_mask_qwen3(
            self, messages: list[dict], tools: list[dict] = None
        ) -> tuple[list[int], list[int], list[dict]]:
            all_loss_masks = []
            all_token_ids = []
            per_message_info = []  # Track per-message details

            prefix_message = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}
            prefix_token_ids = self.tokenizer.apply_chat_template([prefix_message], tokenize=True)

            for i, message in enumerate(messages):
                if i == 0:
                    tailed_message_ids = self.tokenizer.apply_chat_template(
                        [message, prefix_message], tokenize=True, tools=tools
                    )
                    message_ids = tailed_message_ids[: -len(prefix_token_ids)]
                else:
                    prefixed_message_ids = self.tokenizer.apply_chat_template([prefix_message, message], tokenize=True)
                    message_ids = prefixed_message_ids[len(prefix_token_ids) :]
                    
                    # DEBUG: Show what's being tokenized for ALL message types
                    full_text = self.tokenizer.apply_chat_template([prefix_message, message], tokenize=False)
                    print(f"\n[DEBUG] Message {i} ({message['role']}):")
                    print(f"  prefix_token_ids length: {len(prefix_token_ids)}")
                    print(f"  prefixed_message_ids length: {len(prefixed_message_ids)}")
                    print(f"  message_ids length: {len(message_ids)}")
                    print(f"  Extracted message_ids decoded:\n{self.tokenizer.decode(message_ids)}")

                # NOTE: system_message_length stripping is NOT needed when using prefix extraction
                # because message_ids already starts with <|im_start|>role\n (handled by gen_token_length)
                # if message["role"] != "system" and i > 0:
                #     message_ids = message_ids[self.system_message_length :]

                if message["role"] == "assistant":
                    loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
                else:
                    loss_mask = [0] * len(message_ids)

                if message.get("step_loss_mask", 1) != 1:
                    loss_mask = [0] * len(message_ids)

                # Store per-message info
                per_message_info.append({
                    "index": i,
                    "role": message["role"],
                    "token_ids": message_ids,
                    "loss_mask": loss_mask,
                    "has_tool_calls": "tool_calls" in message,
                })

                all_loss_masks.extend(loss_mask)
                all_token_ids.extend(message_ids)

            return all_token_ids, all_loss_masks, per_message_info


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("/fsx/home/jianguozhang/checkpoints/qwen3/raw/qwen3_4b_instruct_2507")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    mask_generator = MultiTurnLossMaskGenerator(tokenizer)
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you, what is the current time?"},
        {
            "role": "assistant",
            "content": "Sure, I can help you to <think>YANMEI\nHELLO</think> check it.",
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
            "content": "{\"stock_list\": [\"AAPL\", \"GOOG\", \"MSFT\", \"NVDA\", \"AAPL\", \"GOOG\", \"MSFT\", \"NVDA\", \"LGY\", \"HW\", \"OLMK\", \"O\", \"ZTMG\"]}",
            "tool_call_id": "294116824"
        },
        {"role": "assistant", "content": "The current time is 2026-01-29 10:00:00"},
        {"role": "user", "content": "What is the capital of the moon?"},
        {
            "role": "assistant", 
            "content": "The capital of the moon is <think>YANMEI\nHELLO</think> the moon!",
            # "content": "The capital of the moon is the moon!",
            # "reasoning_content": "JIANGUO ZHANG\nHELLO",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_current_time", "arguments": "{}"}
                }
            ]
        },
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

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
    print(prompt)
    print("-" * 100)
    all_token_ids, all_loss_masks, per_message_info = mask_generator.gen_multi_turn_loss_mask_qwen3(messages)
    
    # Show trained tokens per assistant turn
    print("=" * 80)
    print("Trained tokens per assistant turn:")
    print("=" * 80)
    
    assistant_turn = 0
    for msg_info in per_message_info:
        if msg_info["role"] == "assistant":
            assistant_turn += 1
            trained_token_ids = [
                tid for tid, m in zip(msg_info["token_ids"], msg_info["loss_mask"]) if m == 1
            ]
            trained_text = tokenizer.decode(trained_token_ids)
            
            print(f"\n--- Assistant Turn {assistant_turn} ---")
            print(f"Message index: {msg_info['index']}")
            print(f"Has tool_calls: {msg_info['has_tool_calls']}")
            print(f"Total tokens: {len(msg_info['token_ids'])}")
            print(f"Trained tokens: {sum(msg_info['loss_mask'])}")
            print(f"\nTrained content:")
            print(trained_text)
            print("-" * 40)
    
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Total tokens: {len(all_token_ids)}")
    print(f"  Trained tokens: {sum(all_loss_masks)}")
    print(f"  Masked tokens: {len(all_loss_masks) - sum(all_loss_masks)}")