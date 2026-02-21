
import os
import sys
import numpy as np
from scene.colmap_loader import read_points3D_binary, read_points3D_text
from scene.dataset_readers import storePly

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python convert_colmap_to_ply.py <input_colmap_sparse_dir> <output_ply_path> [--expand_frames N]")
        return
    
    colmap_path = args[0]
    output_path = args[1]
    
    expand_frames = 1
    if "--expand_frames" in args:
        idx = args.index("--expand_frames")
        if idx + 1 < len(args):
            expand_frames = int(args[idx+1])
            
    bin_path = os.path.join(colmap_path, "points3D.bin")
    txt_path = os.path.join(colmap_path, "points3D.txt")

    if not os.path.exists(bin_path) and not os.path.exists(txt_path):
        # Try checking subdir '0'
        bin_path = os.path.join(colmap_path, "0", "points3D.bin")
        txt_path = os.path.join(colmap_path, "0", "points3D.txt")

    if os.path.exists(bin_path):
        print(f"Reading from {bin_path}")
        xyz, rgb, _ = read_points3D_binary(bin_path)
    elif os.path.exists(txt_path):
        print(f"Reading from {txt_path}")
        xyz, rgb, _ = read_points3D_text(txt_path)
    else:
        print(f"Error: Could not find points3D.bin or points3D.txt in {colmap_path}")
        return

    print(f"Found {xyz.shape[0]} base points.")
    
    if expand_frames > 1:
        print(f"Expanding points to {expand_frames} frames...")
        base_xyz = xyz
        base_rgb = rgb
        
        all_xyz = []
        all_rgb = []
        all_time = []
        
        for i in range(expand_frames):
            t = float(i) / expand_frames if expand_frames > 1 else 0.0
            all_xyz.append(base_xyz)
            all_rgb.append(base_rgb)
            # Create time channel for this frame
            t_channel = np.full((base_xyz.shape[0], 1), t)
            all_time.append(t_channel)
            
        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)
        time = np.concatenate(all_time, axis=0)
        
        print(f"Expanded to {xyz.shape[0]} points (with time).")
        print(f"Writing to {output_path}")
        storePly(output_path, xyz, rgb, time=time)
    else:
        print(f"Writing to {output_path}")
        storePly(output_path, xyz, rgb)
        
    print("Done.")

if __name__ == "__main__":
    main()
