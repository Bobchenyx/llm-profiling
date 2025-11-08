# !pip install huggingface_hub hf_transfer
import os
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

print("Starting download...")

# snapshot_download(
#     repo_id = "deepseek-ai/DeepSeek-V2-Lite",
#     local_dir = "deepseek-ai/DeepSeek-V2-Lite",
# )

snapshot_download(
    repo_id = "deepseek-ai/DeepSeek-V3-0324",
    local_dir = "deepseek-ai/DeepSeek-V3-0324",
)


print("Download finished.")