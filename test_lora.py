import torch
import torch.distributed as dist
from diffsynth import ModelManager, save_video, VideoData
from FixModel import FixPipeline
from PIL import Image
import cv2
import os
import argparse
import glob
from typing import List

def save_img(video: List[Image.Image], output_folder: str, cam_name: str, start_index: int = 0):
    """Saves a list of PIL Images to a directory."""
    images_dir = os.path.join(output_folder, "images")
    os.makedirs(images_dir, exist_ok=True)

    for i, frame in enumerate(video):
        file_path = os.path.join(images_dir, f"{cam_name}_{start_index + i:04d}.png")
        frame.save(file_path)
        print(file_path)

    print(f"Saved {len(video)} images for camera {cam_name} to {images_dir} starting at index {start_index}")

def main():
    parser = argparse.ArgumentParser(description="Generate video from image sequence.")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to folder containing images')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output video folder')
    parser.add_argument('--model_path', type=str, default='./checkpoints/4DSloMo_LoRA.ckpt', help='Path to the LoRA model checkpoint')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--sliding_window_frame', type=int, default=33, help='Number of frames per sliding window chunk')
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    if rank == 0:
        print(f"Running distributed inference on {world_size} GPUs.")

    image_folder = args.input_folder
    output_folder = args.output_folder

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models(
        ["checkpoints/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32,
    )

    model_manager.load_models(
        [
            [
                "checkpoints/diffusion_pytorch_model-00001-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00002-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00003-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00004-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00005-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00006-of-00007.safetensors",
                "checkpoints/diffusion_pytorch_model-00007-of-00007.safetensors",
            ],
            "checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
            "checkpoints/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,
    )
    model_manager.load_lora(args.model_path, lora_alpha=1.0)
    pipe = FixPipeline.from_model_manager(model_manager, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    # Dynamic camera list generation
    gt_crop_dir = os.path.join(image_folder, "test", "ours_None", "gt_crop")
    if not os.path.exists(gt_crop_dir):
         print(f"Warning: {gt_crop_dir} not found. Trying 'ours_30000'...")
         gt_crop_dir = os.path.join(image_folder, "test", "ours_30000", "gt_crop")
         if not os.path.exists(gt_crop_dir):
             raise FileNotFoundError(f"GT crop directory not found in ours_None or ours_30000: {gt_crop_dir}")
    
    video_crop_dir = os.path.join(image_folder, "test", "ours_None", "video_crop")
    mp4_files = glob.glob(os.path.join(video_crop_dir, "*.mp4"))
    if not mp4_files:
        raise ValueError(f"No *.mp4 files found in {video_crop_dir}")
        
    cam_list = []
    for f in mp4_files:
        basename = os.path.basename(f)
        # Assuming format name_0000.png
        cam_name = basename.replace(".mp4", "")
        cam_list.append(cam_name)
    
    cam_list.sort()
    print(f"Found {len(cam_list)} cameras: {cam_list}")

    cameras_for_this_rank = cam_list[rank::world_size]
    print(f"Rank {rank} is assigned {len(cameras_for_this_rank)} cameras.")

    for cam_name in cameras_for_this_rank:
        print(f"------ Rank {rank} processing {cam_name} ------")
        # image_path = f"{image_folder}/test/ours_None/gt_crop/{cam_name}_0000.png"
        # if not os.path.exists(image_path):
        #     raise FileNotFoundError(f"Image not found for camera {cam_name}: {image_path}")
        # try:
        #     image = Image.open(image_path)
        # except Exception as e:
        #     raise RuntimeError(f"Failed to open image for camera {cam_name}: {image_path}") from e

        video_path = f"{image_folder}/test/ours_None/video_crop/{cam_name}.mp4"
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found for camera {cam_name}: {video_path}")

        print(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for camera {cam_name}: {video_path}")

        video = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            video.append(pil_frame)

        cap.release()
        if len(video) == 0:
            raise ValueError(f"No frames could be read from video for camera {cam_name}: {video_path}")

        # --- Sliding Window Logic ---
        total_frames = len(video)
        window_size = args.sliding_window_frame
        
        # Available GT crop images for this camera
        # Typically named cam_name_{frame_index:04d}.png (or just cam_name_{idx}.png)
        # We assume files in gt_crop are like 1930_0000.png, 1930_0001.png etc.
        # We need to find the one closest to our current window start.

        # List all GT frames for this camera
        gt_files = glob.glob(os.path.join(gt_crop_dir, f"{cam_name}_*.png"))
        gt_indices = []
        gt_file_map = {}
        for f in gt_files:
            try:
                # expecting cam_name_XXXX.png. Split by '_' and take last part?
                # filename: 19305323_0000.png -> 19305323, 0000.png
                fname = os.path.basename(f)
                idx_str = fname.replace(f"{cam_name}_", "").split('.')[0]
                idx = int(idx_str)
                gt_indices.append(idx)
                gt_file_map[idx] = f
            except ValueError:
                continue
        gt_indices.sort()
        
        if not gt_indices:
             print(f"Warning: No valid GT files found for camera {cam_name} to search nearest. Using default 0000.")
             # Fallback logic if needed, or error out
        
        for start_idx in range(0, total_frames, window_size):
            end_idx = min(start_idx + window_size, total_frames)
            chunk_len = end_idx - start_idx
            
            # 1. Find nearest GT frame
            # We want an image close to start_idx or center of the chunk? 
            # User said "nearest image". 
            # Let's pick the one closest to start_idx.
            nearest_idx = min(gt_indices, key=lambda x: abs(x - start_idx))
            best_img_path = gt_file_map[nearest_idx]
            print(f"Chunk {start_idx}-{end_idx}: Closest GT image is frame {nearest_idx} ({os.path.basename(best_img_path)})")
            
            try:
                input_cond_image = Image.open(best_img_path)
            except Exception as e:
                print(f"Error opening image {best_img_path}: {e}")
                continue

            # 2. Prepare video chunk
            current_video_chunk = video[start_idx:end_idx]
            
            # 3. Process Chunk
            print(f"Processing chunk frames {start_idx} to {end_idx} ({len(current_video_chunk)} frames)...")
            
            # Ensure num_frames % 4 == 1 if required by model, but we can't change video length arbitrarily.
            # The FixPipeline adjusts num_frames internally or we might need to handle it.
            # For now passing the chunk.
            
            output_chunk = pipe(
                prompt="A girl dancing",
                negative_prompt="Vivid colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static image, cluttered background, three legs, crowded background, walking backwards",
                input_image=input_cond_image,
                input_video=current_video_chunk,
                num_inference_steps=args.num_inference_steps,
                seed=0,
                num_frames=len(current_video_chunk), 
                tiled=True,
                height=args.height, width=args.width, cam_id=0
            )
            
            # 4. Save Chunk
            save_img(output_chunk, output_folder, cam_name, start_index=start_idx)
            
            # Clear memory
            del output_chunk
            torch.cuda.empty_cache()

    dist.destroy_process_group()

if __name__ == '__main__':
    main()