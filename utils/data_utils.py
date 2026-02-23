import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np


def camera_collate_fn(batch):
    """Custom collate: stack GT images normally, keep camera objects as a plain list.
    This is necessary because Camera objects may contain CUDA tensors that
    cannot be stacked / pickled by the default collate.
    """
    images, cameras = zip(*batch)
    images = torch.stack(images, dim=0)   # (B, C, H, W)
    return images, list(cameras)

class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        
    def __getitem__(self, index):
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
            
        return viewpoint_image, viewpoint_cam
    
    def __len__(self):
        return len(self.viewpoint_stack)
    
