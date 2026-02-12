import os
from huggingface_hub import snapshot_download

# 設定下載目標為 "當前目錄下的 checkpoints"
# 因為我們剛剛做了捷徑，這裡實際上會寫入 /work
base_dir = "./checkpoints"

print(f"正在下載模型至: {os.path.abspath(base_dir)}")

# 1. 下載 4DSloMo LoRA
# 根據 README，這會包含 4DSloMo_LoRA.ckpt
# 我們直接載入 base_dir，這樣檔案就會出現在 checkpoints/4DSloMo_LoRA.ckpt
print("Step 1: 下載 4DSloMo LoRA...")
snapshot_download(
    repo_id="yutian05/4DSloMo",
    local_dir=base_dir,
    local_dir_use_symlinks=False
)

# 2. 下載 Wan2.1 (這顆很大！)
# 建議多一層資料夾，避免跟其他檔案混在一起
# 下載後路徑會是: checkpoints/Wan2.1-I2V-14B-720P/
print("Step 2: 下載 Wan2.1 基礎模型...")
snapshot_download(
    repo_id="Wan-AI/Wan2.1-I2V-14B-720P",
    local_dir=base_dir,
    local_dir_use_symlinks=False
)

print("=== 所有模型下載完成 ===")
print(f"請檢查: {base_dir}")