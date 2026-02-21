# 1. 關鍵！必須使用 NVIDIA 的 devel 版本 (包含 nvcc 編譯器)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 2. 設定環境變數，避免安裝時跳出對話框
ENV DEBIAN_FRONTEND=noninteractive

# 3. 安裝系統基礎套件 (包含 OpenCV 需要的 libGL)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 4. 安裝 Miniconda (比 Anaconda 輕量，功能一樣)
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# --- 以下開始完全依照 README 流程 ---

# 5. 建立 Conda 環境 (Python 3.10)
RUN conda create -n 4dslomo -c conda-forge --override-channels python=3.10 -y

# 6. 重要技巧：讓後面的指令都在 conda 環境內執行
SHELL ["conda", "run", "-n", "4dslomo", "/bin/bash", "-c"]

# 7. 安裝 PyTorch (依照 README)
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# 8. 複製您的程式碼到映像檔中
WORKDIR /app
COPY . /app

# 9. 安裝 requirements.txt
# (請確保您的 requirements.txt 裡移除了 ./simple-knn 和 ./pointops2，我們在下一步手動裝)
RUN pip install -r requirements.txt

# 10. 安裝需要編譯的 CUDA 模組 (simple-knn, pointops2)
# 這裡會用到 nvcc，因為我們選對了 Base Image，所以會成功
# 設定 GPU 架構清單 (這行一定要加在編譯 simple-knn 之前)
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# 強制使用系統安裝的 CUDA 編譯器
ENV FORCE_CUDA="1"

# 安裝需要編譯的套件
RUN pip install --no-build-isolation -v ./simple-knn
RUN pip install --no-build-isolation -v ./pointops2

# 11. 【推薦做法】直接將 Conda 環境加入 PATH
# 這樣做不需要 source activate，無論 Singularity 還是 Docker 都能直接抓到對的 Python
ENV PATH="/opt/conda/envs/4dslomo/bin:$PATH"

# 12. (選用) 設定預設 shell 讓進入容器時感覺像是在環境裡
# 這只對互動模式 (singularity shell) 有效，但對執行腳本沒影響 (靠上面那行 PATH 就夠了)
RUN echo "source activate 4dslomo" >> /etc/bash.bashrc