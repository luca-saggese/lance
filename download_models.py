from huggingface_hub import snapshot_download

save_dir = "./downloads/"
repo_id = "bytedance-research/Lance" 
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt","*.pth",],
)