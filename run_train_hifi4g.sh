#!/bin/bash
#SBATCH --job-name=4DSloMo_Train    # 作業名稱
#SBATCH -p normal2            # 使用 gp1d 分割區 (1張 GPU)
#SBATCH --nodes=1                   # 使用 1 個節點
#SBATCH --cpus-per-task=20
#SBATCH –gres=gpu:1           # 使用 1 張 GPU
#SBATCH --mem=60G                   # 記憶體 (4DSloMo 吃記憶體，給大一點)
#SBATCH --time=24:00:00            # 執行時間上限 (24小時)
#SBATCH --output=log_%j.out         # 標準輸出 Log (包含 print 的內容)
#SBATCH --error=err_%j.err          # 錯誤訊息 Log
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --mail-user=a565854525658545256585452@gmail.com
#SBATCH -A ACD115013

# ================= 設定區塊 (請修改這裡) =================
# 1. 您的 SIF 映像檔位置
SIF_IMAGE="/work/$(whoami)/4DSloMo.sif"
WORK_DIR="/work/u9859221/4DSloMo"

# ========================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# 載入 Singularity 模組 (有些節點需要，加了保險)
module load singularity

# 開始訓練
# 關鍵參數解析：
# --nv       : 啟用 NVIDIA GPU 支援 (沒加會找不到 CUDA)
# -B /work   : 將 /work 掛載進去 (讓程式能讀寫您的 Code 和 Data)
# python ... : 執行當前目錄下的 train.py
cd /home/u9859221/4DSloMo

#data
singularity exec --nv -B /work --pwd "$WORK_DIR" /work/$(whoami)/4DSloMo.sif \
    /bin/bash -c '
        hf download moqiyinlun1/HiFiHuman --repo-type dataset --local-dir ./datasets --include "HiFi4G_Dataset/4K_Actor1_Greeting/*"\
        && cat ./datasets/HiFi4G_Dataset/4K_Actor1_Greeting/4K_Actor1_Greeting.zip.part* > ./datasets/4K_Actor1_Greeting.zip \
        && unzip -o ./datasets/4K_Actor1_Greeting.zip -d ./datasets/4K_Actor1_Greeting \
        && rm ./datasets/4K_Actor1_Greeting.zip \
        && cd preprocess \
        && python hifi4g_process.py --input ../datasets/4K_Actor1_Greeting --output ../datasets/4K_Actor1_Greeting_preprocess --move\
        && cd .. \
        && python convert_colmap_to_ply.py ./datasets/4K_Actor1_Greeting/image_white_undistortion/colmap/sparse/0 ./datasets/4K_Actor1_Greeting_preprocess/points3d.ply --expand_frames 200
        '

singularity exec --nv -B /work --pwd "$WORK_DIR" /work/$(whoami)/4DSloMo.sif \
    /bin/bash -c "
        ulimit -n 65535 && \
        python train.py --config ./configs/default.yaml --model_path ./output/4K_Actor1_Greeting_preprocess --source_path ./datasets/4K_Actor1_Greeting_preprocess && \
        python render.py --model_path ./output/4K_Actor1_Greeting_preprocess/ --loaded_pth=./output/4K_Actor1_Greeting_preprocess/chkpnt30000.pth --skip_video --time_duration -0.5 1.5 && \
        python process_video.py --input_folder ./output/4K_Actor1_Greeting_preprocess/test/ours_None/ --max_frames 200 && \
        CUDA_VISIBLE_DEVICES=0  torchrun --nproc_per_node=1 test_lora.py --input_folder ./output/4K_Actor1_Greeting_preprocess --output_folder ./datasets/4K_Actor1_Greeting_preprocess_wan/ --model_path ./checkpoints/4DSloMo_LoRA.ckpt --num_inference_steps 5 --sliding_window_frame 33 --height 1024 --width 1024 && \
        cp ./datasets/4K_Actor1_Greeting_preprocess/transforms_valid.json ./datasets/4K_Actor1_Greeting_preprocess_wan/transforms_test.json && cp ./datasets/4K_Actor1_Greeting_preprocess/transforms_train_stage2.json ./datasets/4K_Actor1_Greeting_preprocess_wan/transforms_train.json && cp ./datasets/4K_Actor1_Greeting_preprocess/points3d.ply ./datasets/4K_Actor1_Greeting_preprocess_wan && \
        python train.py --config ./configs/default_stage2.yaml --model_path ./output/4K_Actor1_Greeting_preprocess_wan --source_path ./datasets/4K_Actor1_Greeting_preprocess_wan
    "

echo "End time: $(date)"