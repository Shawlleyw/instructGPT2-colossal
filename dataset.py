
import utils
import copy
from torch.utils.data import Dataset

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
        
def prepare_dataset(path, tokenizer):
    raw_data = utils.load_json_data(path)
    inputs_text = [utils.polish_by_prompt(record) for record in raw_data]
    # format?: Should response starts with a new line?
    responses_text = [f"{record['output']}{tokenizer.base_tokenizer.eos_token}" for record in raw_data]
    targets_text = [i + r for i, r in zip(inputs_text, responses_text)]
    
    inputs_tokens = [tokenizer(input) for input in inputs_text]
    targets_tokens = [tokenizer(target) for target in targets_text]
    input_ids = targets_tokens
    labels = copy.deepcopy(input_ids)
    input_lens = [input_tokens.ne(tokenizer.base_tokenizer.pad_token_id).sum().item() for input_tokens in inputs_tokens]
    print(input_lens)
    for label, input_len in zip(labels, input_lens):
        label[:input_len] = tokenizer.IGNORE_INDEX
    return InstructDataset(input_ids, labels)
    
    
    