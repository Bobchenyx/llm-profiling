# llm-profiling
LLM Profiling with DeepSpeed Flops Profiler

## Installation

```bash
git clone https://github.com/Bobchenyx/llm-profiling.git
cd llm-profiling

conda create -n llm-profiling python=3.10 -y
conda activate llm-profiling

pip install -r requirements.txt
```

## Usage

If you need to download the model to your local machine, please refer to the `hf.py` script.

```bash
# pip install huggingface_hub hf_transfer
python hf.py
```

Please remember to specify the model path before running the script.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python llm-profiler.py 2>&1 | tee logs/<llm-provider>.log

```

Sample output provided in [`logs/`](./logs/) dir.

If people are interested in `torch.profiler` with DeepSeek MoE, please checkout [Bobchenyx/DeepSeek-V3/tree/llm-profiling](https://github.com/Bobchenyx/DeepSeek-V3/tree/llm-profiling)

---
## Citation

If this work is helpful, please kindly cite as:

```bibtex
@article{chen2025collaborative,
  title={Collaborative Compression for Large-Scale MoE Deployment on Edge},
  author={Chen, Yixiao and Xie, Yanyue and Yang, Ruining and Jiang, Wei and Wang, Wei and He, Yong and Chen, Yue and Zhao, Pu and Wang, Yanzhi},
  journal={arXiv preprint arXiv:2509.25689},
  year={2025}
}
```

## Acknowledgements

This repository builds upon the outstanding work of the following open-source authors and projects:

- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- [Qwen3](https://github.com/QwenLM/Qwen3)

We sincerely thank them for their excellent contributions to the open-source community.
