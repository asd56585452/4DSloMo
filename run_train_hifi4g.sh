#!/bin/bash
#SBATCH --job-name=4DSloMo_Train    # 作業名稱
#SBATCH -p normal            # 使用 gp1d 分割區 (1張 GPU)
#SBATCH --nodes=1                   # 使用 1 個節點
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1           # 使用 1 張 GPU
#SBATCH --mem=200G                   # 記憶體 (4DSloMo 吃記憶體，給大一點)
#SBATCH --time=24:00:00            # 執行時間上限 (24小時)
#SBATCH --output=log_%j.out         # 標準輸出 Log (包含 print 的內容)
#SBATCH --error=err_%j.err          # 錯誤訊息 Log
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --mail-user=a565854525658545256585452@gmail.com
#SBATCH -A ACD115013

# ================= 設定區塊 (請修改這裡) =================
# 1. 您的 SIF 映像檔位置
SIF_IMAGE="/work/$(whoami)/4DSloMo.sif"
WORK_DIR="/home/u9859221/4DSloMo"
MY_TMPDIR="/dev/shm/$(whoami)_$SLURM_JOB_ID"
DATASETS_NAME_1="DualGS_Dataset"
DATASETS_NAME_2="4K_Actor1_Flute"
# ========================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "使用的本機暫存目錄: $MY_TMPDIR"

# 載入 Singularity 模組 (有些節點需要，加了保險)
module load singularity

# 開始訓練
# 關鍵參數解析：
# --nv       : 啟用 NVIDIA GPU 支援 (沒加會找不到 CUDA)
# -B /work   : 將 /work 掛載進去 (讓程式能讀寫您的 Code 和 Data)
# python ... : 執行當前目錄下的 train.py
cd $WORK_DIR

mkdir -p $MY_TMPDIR/datasets
mkdir -p $MY_TMPDIR/output

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1

#data
singularity exec --nv -B /work -B $MY_TMPDIR --pwd "$WORK_DIR" /work/$(whoami)/4DSloMo.sif \
    /bin/bash -c "
        hf download moqiyinlun1/HiFiHuman --repo-type dataset --local-dir $MY_TMPDIR/datasets --include $DATASETS_NAME_1/$DATASETS_NAME_2/*\
        && cat $MY_TMPDIR/datasets/$DATASETS_NAME_1/$DATASETS_NAME_2/$DATASETS_NAME_2.zip.part* > $MY_TMPDIR/datasets/$DATASETS_NAME_2.zip \
        && rm $MY_TMPDIR/datasets/$DATASETS_NAME_1/$DATASETS_NAME_2/$DATASETS_NAME_2.zip.part* \
        && unzip -qo $MY_TMPDIR/datasets/$DATASETS_NAME_2.zip -d $MY_TMPDIR/datasets/$DATASETS_NAME_2 \
        && rm $MY_TMPDIR/datasets/$DATASETS_NAME_2.zip \
        && cd preprocess \
        && python hifi4g_process.py --input $MY_TMPDIR/datasets/$DATASETS_NAME_2 --output $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess --move true \
        && cd .. \
        && python convert_colmap_to_ply.py $MY_TMPDIR/datasets/$DATASETS_NAME_2/image_white_undistortion/colmap/sparse/0 $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess/points3d.ply --expand_frames 200 \
        && ulimit -n 65535 && \
        python train.py --config ./configs/default.yaml --model_path $MY_TMPDIR/output/$DATASETS_NAME_2_preprocess --source_path $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess --preload && \
        python render.py --model_path $MY_TMPDIR/output/$DATASETS_NAME_2_preprocess/ --loaded_pth=$MY_TMPDIR/output/$DATASETS_NAME_2_preprocess/chkpnt43750.pth --skip_video --time_duration -0.5 1.5 && \
        rm -r $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess && \
        python process_video.py --input_folder $MY_TMPDIR/output/$DATASETS_NAME_2_preprocess/test/ours_None/ --max_frames 200 && \
        CUDA_VISIBLE_DEVICES=0  torchrun --nproc_per_node=1 test_lora.py --input_folder $MY_TMPDIR/output/$DATASETS_NAME_2_preprocess --output_folder $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess_wan/ --model_path ./checkpoints/4DSloMo_LoRA.ckpt --num_inference_steps 5 --sliding_window_frame 33 --height 1024 --width 1024 && \
        cp $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess/transforms_valid.json $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess_wan/transforms_test.json && cp $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess/transforms_train_stage2.json $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess_wan/transforms_train.json && cp $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess/points3d.ply $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess_wan && \
        rm -r $MY_TMPDIR/output/$DATASETS_NAME_2_preprocess && \
        python train.py --config ./configs/default_stage2.yaml --model_path ./output/$DATASETS_NAME_2_preprocess_wan --source_path $MY_TMPDIR/datasets/$DATASETS_NAME_2_preprocess_wan --preload && \
        python render.py --model_path ./output/$DATASETS_NAME_2_preprocess_wan/ --loaded_pth=./output/$DATASETS_NAME_2_preprocess_wan/chkpnt43750.pth --skip_video --time_duration -0.5 1.5
    "
rm -rf $MY_TMPDIR
echo "End time: $(date)"