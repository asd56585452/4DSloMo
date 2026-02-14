import os
import argparse
import natsort
from tqdm import tqdm
import shutil

import json
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process HIFI4G data')
    parser.add_argument('--input', type=str, required=True, help='Path to the input data directory')
    parser.add_argument('--output', type=str, required=True, help='Path to the output data directory')
    parser.add_argument('--move', type=bool, default=False, help='If move the original data to the target folder')
    parser.add_argument('--fps_scale', type=int, default=4, help='Number of frames to delay')
    parser.add_argument('--valid_ratio', type=int, default=8, help='Ratio of valid frames')
    args = parser.parse_args()

    # assert input and output are same
    assert args.input != args.output, 'Input and output directories are same'

    if not os.path.exists(args.input):
        raise ValueError('Input directory does not exist')
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # generate transforms.json
    text_path = os.path.join(args.input, 'colmap', 'sparse')
    output_json_path = os.path.join(args.output, 'transforms.json')
    colmap2k_cmd = f"python colmap2k.py --text {text_path} --out {output_json_path} --keep_colmap_coords"
    os.system(colmap2k_cmd)

    with open(output_json_path, 'r') as f:
        base_transforms = json.load(f)

    # move the data
    images_folder_path = os.path.join(args.input, 'image_undistortion_white')
    frames = os.listdir(images_folder_path)
    frames = natsort.natsorted(frames)

    output_images_dir = os.path.join(args.output, 'images')
    os.makedirs(output_images_dir, exist_ok=True)

    all_frames = []

    for cam_frame in base_transforms['frames']:
        for i, frame in enumerate(frames):
            frame_source_path = os.path.join(images_folder_path, frame)
            # Assumes frame_source_path contains images directly (e.g. cam1.png, cam2.png)
            
            # Calculate time (normalized 0.0 to 1.0)
            time_val = float(i) / (len(frames)) if len(frames) > 1 else 0.0

        # for cam_frame in base_transforms['frames']:
            # Example file_path in base: "images/19224108.png" or "images/19224108"
            # We want to extract "19224108"
            base_file_path = cam_frame['file_path']
            basename = os.path.basename(base_file_path)
            cam_name = os.path.splitext(basename)[0]
            extension = os.path.splitext(basename)[1]
            if not extension:
                extension = ".png" # optimize for the user request context if missing

            src_img_name = cam_name + extension
            src_img_path = os.path.join(frame_source_path, src_img_name)

            if not os.path.exists(src_img_path):
                # Try finding without extension if base had none, or verify what's in the folder
                # But for now assume consistency with colmap output
                pass

            new_img_name = f"{int(cam_name):04d}_{i:04d}"
            tgt_img_path = os.path.join(output_images_dir, new_img_name)

            # if args.move:
            #     shutil.move(src_img_path, tgt_img_path)
            # else:
            #     shutil.copy(src_img_path, tgt_img_path)

            new_frame = copy.deepcopy(cam_frame)
            new_frame['file_path'] = f"images/{new_img_name}"
            new_frame['time'] = time_val
            new_frame['frame'] = i
            all_frames.append(new_frame)

    first_frames = []
    valid_camera_names = []
    camera_count = 0
    FPSdelay_split = {"file_path":"FPSdelay"}
    FPSdelay_split_count = 0
    for i, frame in enumerate(tqdm(all_frames)):
        if frame['time'] == 0:
            first_frames.append(frame)
            if camera_count%args.valid_ratio==0:
                valid_camera_names.append(frame['file_path'].split('_')[0])
            else:
                FPSdelay_split[frame['file_path'].split('_')[0]] = FPSdelay_split_count
                FPSdelay_split_count +=1
                FPSdelay_split_count = FPSdelay_split_count%args.fps_scale
            camera_count +=1

    print(valid_camera_names)
    valid_frames = []
    train_stage2_frames = []
    for i, frame in enumerate(tqdm(all_frames)):
        if frame['file_path'].split('_')[0] in valid_camera_names:
            valid_frames.append(frame)
        else:
            train_stage2_frames.append(frame)
    
    train_frames = []
    test_frames = []
    for i, frame in enumerate(tqdm(train_stage2_frames)):
        if FPSdelay_split[frame['file_path'].split('_')[0]]==frame['frame']%args.fps_scale:
            train_frames.append(frame)
        copy_frame = copy.deepcopy(frame)
        copy_frame['file_path'] = copy_frame['file_path'].split('_')[0]+"_0000"
        test_frames.append(copy_frame)

    

    
    # Update and save combined transforms.json
    first_transforms = copy.deepcopy(base_transforms)
    first_transforms['frames'] = first_frames
    
    with open(output_json_path, 'w') as f:
        json.dump(first_transforms, f, indent=4)
    
    valid_json_path = os.path.join(args.output, 'transforms_valid.json')
    valid_transforms = copy.deepcopy(base_transforms)
    valid_transforms['frames'] = valid_frames
    with open(valid_json_path, 'w') as f:
        json.dump(valid_transforms, f, indent=4)
    
    # train_stage2_json_path = os.path.join(args.output, 'transforms_train_stage2.json')
    # train_stage2_transforms = copy.deepcopy(base_transforms)
    # train_stage2_transforms['frames'] = train_stage2_frames
    # with open(train_stage2_json_path, 'w') as f:
    #     json.dump(train_stage2_transforms, f, indent=4)

    train_json_path = os.path.join(args.output, 'transforms_train.json')
    train_transforms = copy.deepcopy(base_transforms)
    train_transforms['frames'] = train_frames
    with open(train_json_path, 'w') as f:
        json.dump(train_transforms, f, indent=4)
    
    test_json_path = os.path.join(args.output, 'transforms_test.json')
    test_transforms = copy.deepcopy(base_transforms)
    test_transforms['frames'] = test_frames
    with open(test_json_path, 'w') as f:
        json.dump(test_transforms, f, indent=4)
    
    
    

