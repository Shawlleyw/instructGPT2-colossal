
import utils
import copy
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
class InstructDataset(Dataset):
    def __init__(self, input_ids, labels):
        assert(len(input_ids) == len(labels))
        self.size = len(input_ids)
        self.input_ids = input_ids
        self.labels = labels
    
    def __len__(self):
        return self.size

    def  __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
class DataCollator:
    
    def __init__(self, input_pad, label_pad):
        self.input_pad = input_pad
        self.label_pad = label_pad
    
    def __call__(self, batch):
        input_ids = [record["input_ids"] for record in batch]
        labels = [record["labels"] for record in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.input_pad)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.label_pad)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.input_pad))
        
def prepare_dataset(path, tokenizer):
    raw_data = utils.load_json_data(path)
    inputs_text = [utils.polish_by_prompt(record) for record in raw_data]
    # format?: Should response starts with a new line?
    responses_text = [f"{record['output']}{tokenizer.base_tokenizer.eos_token}" for record in raw_data]
    targets_text = [i + r for i, r in zip(inputs_text, responses_text)]
    
    with tqdm(inputs_text, desc="Tokenize inputs", ncols=80) as inputs_ts:
        inputs_tokens = [tokenizer(input) for input in inputs_ts]
        
    with tqdm(targets_text, desc="Tokenize targets", ncols=80) as targets_ts:
        targets_tokens = [tokenizer(target) for target in targets_ts]
         
    input_ids = targets_tokens
    labels = copy.deepcopy(input_ids)
    
    
    input_lens = [input_tokens.ne(tokenizer.base_tokenizer.pad_token_id).sum().item() for input_tokens in inputs_tokens]
    for label, input_len in zip(labels, input_lens):
        label[:input_len] = tokenizer.IGNORE_INDEX
    return InstructDataset(input_ids, labels)

