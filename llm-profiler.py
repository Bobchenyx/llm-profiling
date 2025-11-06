from functools import partial
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator


def ds_v2lite_input_constructor(batch_size, seq_len, tokenizer, device):
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


with get_accelerator().device(0):
    device = get_accelerator().current_device()
    device_name = get_accelerator().current_device_name()

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

    kwargs = ds_v2lite_input_constructor(batch_size, seq_len, tokenizer, device)
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


# ====== Bert Example =====

# from functools import partial
# import torch
# from transformers import BertForSequenceClassification, BertTokenizer
# from deepspeed.profiling.flops_profiler import get_model_profile
# from deepspeed.accelerator import get_accelerator


# def bert_input_constructor(batch_size, seq_len, tokenizer):
#     fake_seq = ""
#     for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
#       fake_seq += tokenizer.pad_token
#     inputs = tokenizer([fake_seq] * batch_size,
#                        padding=True,
#                        truncation=True,
#                        return_tensors="pt")
#     labels = torch.tensor([1] * batch_size)
#     inputs = dict(inputs)
#     inputs.update({"labels": labels})
#     return inputs


# with get_accelerator().device(0):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
#     batch_size = 4
#     seq_len = 128
#     enable_profile = True
#     if enable_profile:
#       flops, macs, params = get_model_profile(
#           model,
#           kwargs=bert_input_constructor(batch_size, seq_len, tokenizer),
#           print_profile=True,
#           detailed=True,
#       )
#     else:
#       inputs = bert_input_constructor((batch_size, seq_len), tokenizer)
#       outputs = model(inputs)