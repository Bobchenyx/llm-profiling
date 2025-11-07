# ====== Bert Example =====
# Reference:
# https://www.deepspeed.ai/tutorials/flops-profiler/
# https://deepspeed.readthedocs.io/en/latest/flops-profiler.html
# https://github.com/deepspeedai/DeepSpeed/tree/master/deepspeed/profiling/flops_profiler


from functools import partial
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator


def bert_input_constructor(batch_size, seq_len, tokenizer):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * batch_size)
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


with get_accelerator().device(0):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    batch_size = 4
    seq_len = 128
    enable_profile = True
    if enable_profile:
      flops, macs, params = get_model_profile(
          model,
          kwargs=bert_input_constructor(batch_size, seq_len, tokenizer),
          print_profile=True,
          detailed=True,
      )
    else:
      inputs = bert_input_constructor((batch_size, seq_len), tokenizer)
      outputs = model(inputs)