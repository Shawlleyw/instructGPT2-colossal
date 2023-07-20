# instructGPT2-ColossalAI

This is a language model finetune script in ColossalAI framework.
The pre-trained model is GPT2. The dataset is from alpaca.

## Run
To run on a single GPU, use
```
colossalai run --nproc_per_node=1 train.py -d alpaca_data.json
```