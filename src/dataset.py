
import utils
import copy
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
class InstructDataset(Dataset):
    def __init__(self, input_texts, target_texts):
        assert(len(input_texts) == len(target_texts))
        self.size = len(input_texts)
        self.input_texts = input_texts
        self.target_texts = target_texts
    
    def __len__(self):
        return self.size

    def  __getitem__(self, i):
        return dict(input_text=self.input_texts[i], target_text=self.target_texts[i])
    
class DataCollator:
    
    def __init__(self, input_pad, label_pad, tokenizer):
        self.input_pad = input_pad
        self.label_pad = label_pad
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        input_texts = [record["input_text"] for record in batch]
        target_texts = [record["target_text"] for record in batch]
        
        input_tokens = self.tokenizer(input_texts)
        target_tokens = self.tokenizer(target_texts)
        
        input_ids = target_tokens
        labels = copy.deepcopy(input_ids)
        
        input_lens = [input_token.ne(self.input_pad).sum().item() for input_token in input_tokens]
        for label, input_len in zip(labels, input_lens):
            label[:input_len] = self.label_pad
        
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.input_pad))
        
def prepare_dataset(path, tokenizer):
    raw_data = utils.load_json_data(path)
    input_texts = [utils.polish_by_prompt(record) for record in raw_data]
    # format?: Should response starts with a new line?
    response_texts = [f"{record['output']}{tokenizer.base_tokenizer.eos_token}" for record in raw_data]
    target_texts = [i + r for i, r in zip(input_texts, response_texts)]    
    return InstructDataset(input_texts, target_texts)

