from transformers import AutoTokenizer
from taskstudio.utils.common import open_json, save_json

model_name = "qwen3_4b_instruct_2507"
chat_template_name = "qwen3---4b--30b-a3b--235b-a22b---instruct-2507---xlam-nothink-01292026.jinja"

model_name = "qwen3_4b_thinking_2507"
chat_template_name = "qwen3-4b-thinking-2507---xlam.jinja"

tokenizer_file = open_json("/fsx/home/jianguozhang/checkpoints/qwen3/raw/" + model_name + "/tokenizer_config.json") 
tokenizer_file["chat_template"] = open(chat_template_name).read()
print(tokenizer_file["chat_template"])
save_json("/fsx/home/jianguozhang/checkpoints/qwen3/raw/" + model_name + "/tokenizer_config.json", tokenizer_file)
print("done for model: ", model_name)