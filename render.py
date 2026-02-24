#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
# TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX" pip install ./diff-gaussian-rasterization
# TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX" pip install ./simple-knn/
import time
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from torch.utils.data import DataLoader
from utils.data_utils import camera_collate_fn
# from rembg import remove
import numpy as np
from PIL import Image
import pdb
import imageio

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from concurrent.futures import ThreadPoolExecutor

def render_set(model_path, name, iteration, loader, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    frames = []
    gts = []
    global_idx = 0
    
    # Initialize a ThreadPoolExecutor for background disk I/O
    io_executor = ThreadPoolExecutor(max_workers=8)

    t4 = time.time()
    for gt_images, view_cams in tqdm(loader, desc="Rendering progress"):
        # gt_images: (B, C, H, W) CPU tensor from DataLoader workers
        # view_cams: list of Camera objects (length B)
        for b in range(len(view_cams)):
            idx = global_idx
            view_cam = view_cams[b]
            torch.cuda.synchronize()  # 確保之前的操作都完成
            t0 = time.time()

            # 1. 測量 GPU 渲染時間
            rendering_torch = render(view_cam.cuda(), gaussians, pipeline, background)["render"]
            torch.cuda.synchronize()  # 等待渲染完成
            t1 = time.time()

            # 2. 測量 CPU 記憶體轉換時間
            gt = gt_images[b][0:3, :, :]
            gt_numpy = gt.permute(1, 2, 0).cpu().numpy()
            if not args.skip_video:
                rendering = rendering_torch.permute(1, 2, 0).cpu().numpy()
                frames.append(rendering)
                gts.append(gt_numpy)
            t2 = time.time()

            # 3. 測量硬碟 I/O (存圖) 時間 - 改交給背景執行緒
            image_name = view_cam.image_path.split('/')[-1].split('.')[0]
            io_executor.submit(
                torchvision.utils.save_image, 
                rendering_torch.clone().cpu(), 
                os.path.join(render_path, image_name + f"_{idx:05d}" + ".png")
            )
            io_executor.submit(
                torchvision.utils.save_image, 
                gt.clone().cpu(), 
                os.path.join(gts_path, image_name + ".png")
            )
            t3 = time.time()

            # 印出前幾個 iteration 的時間分佈
            if idx < 50:
                print(background)
                print(f"\n[Iter {idx}] Loader: {t0-t4:.4f}s | Render: {t1-t0:.4f}s | GPU->CPU: {t2-t1:.4f}s | Disk I/O (Async submit): {t3-t2:.4f}s")
            t4 = time.time()
            global_idx += 1
            
    # Wait for all asynchronous disk I/O to finish before making videos
    print("Waiting for background image saving to complete...")
    io_executor.shutdown(wait=True)
    
    if not args.skip_video:
        imageio.mimsave(render_path+'video.mp4', [frame for frame in frames], fps=25)
        imageio.mimsave(render_path+'-gt.mp4', [frame for frame in gts], fps=25)
        print(render_path+'video.mp4')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        scene = Scene(dataset, gaussians, time_duration=args.time_duration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        bg_color = [1,1,1] 
        # bg_color = [0.125,0.216,0.157]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # DataLoader 設定：num_workers 控制平行讀圖的 CPU 子進程數
        # pin_memory=True 可加速 CPU→GPU 的資料傳輸
        num_workers = args.num_workers if hasattr(args, 'num_workers') else 11

        # if not skip_train:
        #     train_loader = DataLoader(scene.getTrainCameras(), batch_size=1,
        #                               num_workers=num_workers, pin_memory=True,
        #                               collate_fn=camera_collate_fn)
        #     render_set(dataset.model_path, "train", scene.loaded_iter, train_loader, gaussians, pipeline, background)

        if not skip_test:
            test_loader = DataLoader(
                scene.getTestCameras(),
                batch_size=1,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=camera_collate_fn,
            )
            render_set(dataset.model_path, "test", scene.loaded_iter, test_loader, gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of DataLoader worker processes for parallel image loading")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 1.5])
    parser.add_argument("--skip_video", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)