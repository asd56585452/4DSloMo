from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np
from pathlib import Path

def readImages(renders_dir, gt_dir):
    renders_dir = Path(renders_dir)
    gt_dir = Path(gt_dir)
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        # GT files don't have the _<idx> suffix, so we strip it off
        # fname is of form "image_name_idx.png"
        gt_fname = fname.rsplit('_', 1)[0] + ".png"
        gt = Image.open(gt_dir / gt_fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def _run_average_metrics(renders, gts, lpips_fn):
    ssims = []
    psnrs = []
    lpipss = []

    for idx in range(len(renders)):
        render_rescaled = renders[idx] * 2.0 - 1.0
        gt_rescaled = gts[idx] * 2.0 - 1.0
        
        ssims.append(ssim(renders[idx], gts[idx]).item())
        psnrs.append(psnr(renders[idx], gts[idx]).item())
        lpipss.append(lpips_fn(render_rescaled, gt_rescaled).mean().item())

    return np.mean(ssims), np.mean(psnrs), np.mean(lpipss)
    
def evaluate(eval_path):
    print(f"---Evaluating: {eval_path}---")
    
    # Check if renders/gt are directly inside eval_path or in an 'ours_X' subfolder
    renders_dir = os.path.join(eval_path, "renders")
    gt_dir = os.path.join(eval_path, "gt")
    
    if not os.path.exists(renders_dir) or not os.path.exists(gt_dir):
        # Look for an 'ours_X' folder (like ours_30000)
        methods = [m for m in os.listdir(eval_path) if m.startswith("ours_")]
        if methods:
            # use the first one found or sort them based on your needs
            method = sorted(methods)[-1] 
            print(f"Using subfolder: {method}")
            renders_dir = os.path.join(eval_path, method, "renders")
            gt_dir = os.path.join(eval_path, method, "gt")
        else:
            print(f"Error: Could not find 'renders' and 'gt' directories in {eval_path}")
            return

    renders, gts, _ = readImages(renders_dir, gt_dir)
    
    if renders:
        avg_ssim, avg_psnr, avg_lpips = _run_average_metrics(renders, gts, lpips_fn)
        
        print(f"  Avg SSIM : {avg_ssim:>12.7f}")
        print(f"  Avg PSNR : {avg_psnr:>12.7f}")
        print(f"  Avg LPIPS: {avg_lpips:>12.7f}")
                    
        results_dict = {
            "SSIM": avg_ssim, 
            "PSNR": avg_psnr, 
            "LPIPS": avg_lpips
        }
        
        output_json = os.path.join(eval_path, "results.json")
        # Try to update existing or create new
        if os.path.exists(output_json):
            try:
                with open(output_json, 'r') as fp:
                    existing_data = json.load(fp)
                    if isinstance(existing_data, dict):
                        results_dict = {**existing_data, **results_dict}
            except Exception:
                pass

        with open(output_json, 'w') as fp:
            json.dump(results_dict, fp, indent=4)
        print(f"Results saved to {output_json}")
    else:
        print("No valid images found to evaluate.")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--eval_path', '-p', required=True, type=str, help="Path to evaluation directory containing renders and gt folders")
    args = parser.parse_args()
    
    evaluate(args.eval_path)