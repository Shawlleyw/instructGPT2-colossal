import json
import torch


prompt_input = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)
prompt_no_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def polish_by_prompt(record):
    global prompt_input, prompt_no_input
    return prompt_input.format_map(record) if ("input" in record) else prompt_no_input.format_map(record)

def print_memory_usage(prefix):
    def toGB(mem):
        return mem / 1024 / 1024 / 1024
    used_mem = torch.cuda.memory_allocated()
    reserved_mem = torch.cuda.memory_reserved()
    max_mem = torch.cuda.max_memory_allocated()
    print(prefix+"memory usage: {0:.2f}GB/{1:.2f}GB, max: {2:.2f}GB".format(toGB(used_mem), toGB(used_mem + reserved_mem), toGB(max_mem)))