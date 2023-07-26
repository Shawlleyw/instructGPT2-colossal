# instructGPT2-ColossalAI

This is a language model finetune script in ColossalAI framework.
The pre-trained model is GPT2 with 124M parameters. The dataset is from alpaca.

There are 3 different version: Plain TorchDDP, LoRA and QLoRA.
## Run
To run on a single GPU, use
```
colossalai run --nproc_per_node=1 finetune.py -d alpaca_data.json
```

## Experiments

All these experiments were did on a V100S-16Q with 16GB GRAM.

#### Plain TorchDDP


```sh
Before training: memory usage: 0.94GB/2.40GB, max: 0.94GB
epoch: 1 / 1: 100%|██████████| 13000/13000 [18:18<00:00, 11.84it/s, loss=1.18]  
After training: memory usage: 2.34GB/12.94GB, max: 7.47GB
```

Size of adpter_model: 475MB

#### LoRA

Introduce peft for LoRA.

Trainable Parameters: 0.3M

```sh
trainable params: 294912 || all params: 124737792 || trainable%: 0.2364255413467636
Before training: memory usage: 0.48GB/1.47GB, max: 0.94GB
epoch: 1 / 1: 100%|██████████| 13000/13000 [13:28<00:00, 16.07it/s, loss=8.01]
After training: memory usage: 0.48GB/8.53GB, max: 5.19GB
```

Size of adpter_model: 1.2MB

#### QLoRA 8-bit

Introduce quantization by loading pre-trained model in 8bit.

Trainable Parameters: 0.3M

```sh
trainable params: 294912 || all params: 124737792 || trainable%: 0.2364255413467636
Before training: memory usage: 0.25GB/0.90GB, max: 0.48GB
epoch: 1 / 1: 100%|██████████| 13000/13000 [12:54<00:00, 16.79it/s, loss=8.53]
After training: memory usage: 0.26GB/3.83GB, max: 2.75GB
```

Size of adpter_model: 1.2MB

## Memory Analysis

#### Models

Normally a model with 124MB parameters takes ~0.48GB memory. However according to [torch forum](https://discuss.pytorch.org/t/memory-consumption-for-the-model-get-doubled-after-wrapped-with-ddp/130837/4), torchDDP plugin allocates extra memory for each parameter for result reduction. This is why the model memory usage(0.94GB) seems doubled before training. After traininng, the gradient is of the same size with parameters and optimizer state is double the size. All these data sum up to be 5 times, and that is ~2.37GB.

From Plain to LoRA, peft freezed pre-trained parameters and add some extra parameters(~0.3M) for training. And the memory usage of these extra parameters as well as their footprints(ddp buffer, gradients, optimizer state) could be ignored. So LoRA model usage should be near the pre-trained model size, which is ~0.48GB.

The same is true for QLoRA. But [8-bit pre-trained model cut down the memory usage by half](https://huggingface.co/docs/transformers/main_classes/quantization#load-a-large-model-in-8bit:~:text=You%20can%20load%20a%20model%20by%20roughly%20halving%20the%20memory%20requirements%20by%20using%20load_in_8bit%3DTrue%20argument%20when%20calling%20.from_pretrained%20method). So QLoRA model should use ~0.24GB memory. 

#### Activations

The memory usage of activations could be $ max_mem - post_trainig_mem $. That's about 5GB for Plain and LoRA and 2.5GB for QLoRA.

The biggest batch results in the same memory usage for Plain and LoRA, while a half for QLoRA due to quantization.