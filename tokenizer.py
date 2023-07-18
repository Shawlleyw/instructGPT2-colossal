import transformers

class Tokenizer:
    
    IGNORE_INDEX = -100
    
    special_tokens = {
        "pad_token": "[PAD]",
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>"
    }
    
    def __init__(self, pretrained_model, max_tokens=512):
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model,
            model_max_length = max_tokens,
            padding_side="right",
        )
        self.num_new_tokens = self.base_tokenizer.add_special_tokens(self.special_tokens)
        print(">>> Tokenizer is loaded")

    def resize_model(self, model):
        model.resize_token_embeddings(len(self.base_tokenizer))
        # initialize new tokens
        if self.num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-self.num_new_tokens:] = input_embeddings_avg
            output_embeddings[-self.num_new_tokens:] = output_embeddings_avg
    
    def __call__(self, input_str):
        assert(type(input_str) == str) # process one seq each time
        tokenized = self.base_tokenizer(
            input_str,
            return_tensors="pt",
            padding="longest",
            max_length=self.base_tokenizer.model_max_length,
            truncation=True,
        )
        return tokenized.input_ids[0]