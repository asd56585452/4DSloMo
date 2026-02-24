import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def camera_collate_fn(batch):
    """Custom collate: stack GT images normally, keep camera objects as a plain list.
    This is necessary because Camera objects may contain CUDA tensors that
    cannot be stacked / pickled by the default collate.
    """
    images, cameras = zip(*batch)
    images = torch.stack(images, dim=0)   # (B, C, H, W)
    return images, list(cameras)

class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack, white_background, preload=False, preload_workers=11):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        self.preloaded_images = None

        if preload:
            print(f"[CameraDataset] Preloading {len(viewpoint_stack)} images into RAM with {preload_workers} workers...")
            
            # Use PyTorch DataLoader for multiprocessing to bypass the GIL
            temp_dataset = CameraDataset(viewpoint_stack, white_background, preload=False)
            
            def preload_collate_fn(batch):
                return [b[0] for b in batch]
                
            dataloader = DataLoader(
                temp_dataset, 
                batch_size=4, # batching reduces IPC overhead
                num_workers=preload_workers, 
                collate_fn=preload_collate_fn, 
                shuffle=False,
                pin_memory=False
            )
            
            self.preloaded_images = []
            for images in tqdm(dataloader, total=len(dataloader), desc="Preloading images"):
                # .clone() allocates a new tensor in the main process memory,
                # disconnecting it from worker shared memory and preventing "Too many open files" errors.
                self.preloaded_images.extend([img.clone() for img in images])
                
            print("[CameraDataset] Preload complete.")

    def _load_image(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only:
            with Image.open(viewpoint_cam.image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
            image_load = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")
            resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
            viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                viewpoint_image *= gt_alpha_mask
            else:
                viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
        else:
            viewpoint_image = viewpoint_cam.image
        return viewpoint_image

    def __getitem__(self, index):
        if self.preloaded_images is not None:
            return self.preloaded_images[index], self.viewpoint_stack[index]
        return self._load_image(index), self.viewpoint_stack[index]
    
    def __len__(self):
        return len(self.viewpoint_stack)
    
