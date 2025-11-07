from functools import partial
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator


def llm_input_constructor(batch_size, seq_len, tokenizer, device):
    # FLOPs-only dummy batch
    # if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    #     tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = 0
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():

    model_name = "./Qwen/Qwen3-30B-A3B-Instruct-2507"   # "./deepseek-ai/DeepSeek-V2-Lite-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # dtype / torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # KV-cache / DynamicCache API 
    model.config.use_cache = False

    batch_size = 4
    seq_len = 128
    enable_profile = True

    first_device = next(model.parameters()).device
    kwargs = llm_input_constructor(batch_size, seq_len, tokenizer, first_device)
    # use_cache=False
    kwargs["use_cache"] = False

    if enable_profile:
        with torch.no_grad(): 
            flops, macs, params = get_model_profile(
                model,
                kwargs=kwargs,
                print_profile=True,
                detailed=True,
                warm_up=1,
                # as_string=False,
            )
        print("FLOPs:", flops, "MACs:", macs, "Params:", params)
    else:
        outputs = model(**kwargs)
        print(outputs.loss.item())
        

if __name__ == "__main__":
    main()