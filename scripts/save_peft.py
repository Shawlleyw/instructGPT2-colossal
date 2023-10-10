from transformers import AutoModelForCausalLM
import argparse


llama2_models = {
    "7b": "meta-llama/Llama-2-7b-chat-hf",
    "70b": "meta-llama/Llama-2-70b-chat-hf",
}

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--save", dest="save_path")
    
    args = parser.parse()
    
    if args.model in llama2_models:
        args.model = llama2_models[args.model]
    
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.load_adapter(args.adapter)
    
    model.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()