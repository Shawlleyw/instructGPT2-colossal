import argparse
import dataset
from tokenizer import Tokenizer

import transformers
from transformers import get_linear_schedule_with_warmup

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

from tqdm import tqdm

PRETRAIN_MODEL = "GPT2"
LEARNING_RATE = 2.4e-5
EPOCHS = 3
WARMUP_RATE = 0.1

def train_epoch(epoch, model, optimizer, lr_sched, dataloader, booster, coord):
    model.train()
    
    with tqdm(dataloader, desc=f"epoch: {epoch + 1} / {EPOCHS}", disable=not coord.is_master()) as pbar:
        for batch in pbar:
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()
            lr_sched.step()
            
            pbar.set_postfix({'loss': loss.item()})
            

def train():
    parser = argparse.ArgumentParser (
        prog="instructGPT2",
        description="fine-tune GPT2 in colossal-AI framework"
    )    
    parser.add_argument("-b", type=int, dest="batch_size", default=4, help="specify batch size")
    parser.add_argument("-d", type=str, dest="data_path", required=True, help="specify data path")
    parser.add_argument("-t", type=int, dest="max_tokens", default=512, help="specify max tokens")
    parser.add_argument("-p", type=str, dest="plugin", default="torch_ddp", choices=["torch_ddp", 'gemini', 'low_level_zero'], help="specify a plugin")
    
    args = parser.parse_args()
    
    colossalai.launch_from_torch(config={}, seed=42)
    coordinator = DistCoordinator()
    
    lr = LEARNING_RATE * coordinator.world_size
    
    if args.plugin == "torch_ddp":
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(placement_policy='cuda', strict_ddp_mode=True, initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
        
    booster = Booster(plugin=plugin)
    
    tokenizer = Tokenizer(PRETRAIN_MODEL, args.max_tokens)
    
    data = dataset.prepare_dataset(args.data_path, tokenizer)
    collator = dataset.DataCollator(
        input_pad=tokenizer.base_tokenizer.pad_token_id, 
        label_pad=tokenizer.IGNORE_INDEX, 
        tokenizer=tokenizer,
    )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(PRETRAIN_MODEL)
    tokenizer.resize_model(model)
    
    dataloader = plugin.prepare_dataloader(data, batch_size=args.batch_size, collate_fn=collator, shuffle=True, drop_last=True)
        
    optimizer = HybridAdam(model.parameters(), lr=lr)
    total_steps = len(dataloader) * EPOCHS
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = WARMUP_RATE * total_steps, 
        num_training_steps = total_steps,
    )
    model, optimizer, _, _, lr_scheduler = booster.boost(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    
    for epoch in range(EPOCHS):
        train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator)
    # booster.save_model(model, "./ckpt/model.pth")


if __name__ == '__main__':
    train()

