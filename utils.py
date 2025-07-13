"""
#################################
Util functions such as
    1) Playing video
    2) Getting FPS
    3) Extracting Frames
    4) Resizing frames
    5) Renaming files of a directory
#################################
"""

#########################################################
# import libraries

import os
import re
import cv2
from config import new_size, server_name
import numpy as np
import tensorflow as tf


#########################################################
# Function definition

def get_paths(server_name):
    """
    Get all paths based on server configuration.
    
    Args:
        server_name: "autodl" or "lambda"
    
    Returns:
        Dictionary containing all paths for the specified server
    """
    if server_name == "autodl":
        paths = {
            # Data paths
            'dir_images': "/root/autodl-tmp/data/Images",
            'dir_masks': "/root/autodl-tmp/data/Masks", 
            'dir_merges': "/root/autodl-tmp/merged_resize/1280to512",
            
            # Output paths with dynamic server_name prefix
            'base_output': f"/root/autodl-tmp/seg_output/{server_name}_Output",
            'model_fig_file': f"/root/autodl-tmp/seg_output/{server_name}_Output/Model_figure/segmentation_model_u_net.png",
            'checkpoint': f"/root/autodl-tmp/seg_output/{server_name}_Output/Models/FireSegmentation.h5",
            
            # # Classification model paths
            # 'classification_models': f"/root/autodl-tmp/{server_name}_Output/Models/",
            # 'classification_h5_models': f"/root/autodl-tmp/{server_name}_Output/Models/h5model/",
            
            # Plot output base
            'plot_base': "/root/autodl-tmp/seg_output/",
            
            # Figure output paths
            'figure_output': f"/root/autodl-tmp/seg_output/{server_name}_Output/Figures/",
            'figure_object': f"/root/autodl-tmp/seg_output/{server_name}_Output/FigureObject/"
        }
    elif server_name == "lambda":
        paths = {
            # Data paths
            'dir_images': "/lambda/nfs/fireseg/data/Images",
            'dir_masks': "/lambda/nfs/fireseg/data/Masks",
            'dir_merges': "/lambda/nfs/fireseg/data/512x512merged",
            
            # Output paths with dynamic server_name prefix
            'base_output': f"/lambda/nfs/fireseg/seg_output/{server_name}_Output",
            'model_fig_file': f"/lambda/nfs/fireseg/seg_output/{server_name}_Output/segmentation_model_u_net.png",
            'checkpoint': f"/lambda/nfs/fireseg/seg_output/{server_name}_Output/model_checkpoints/FireSegmentation.h5",
            
            # # Classification model paths
            # 'classification_models': f"/lambda/nfs/fireseg/seg_output/{server_name}_Output/Models/",
            # 'classification_h5_models': f"/lambda/nfs/fireseg/seg_output/{server_name}_Output/Models/h5model/",
            
            # Plot output base
            'plot_base': "/lambda/nfs/fireseg/seg_output/",
            
            # Figure output paths
            'figure_output': f"/lambda/nfs/fireseg/seg_output/{server_name}_Output/Figures/",
            'figure_object': f"/lambda/nfs/fireseg/seg_output/{server_name}_Output/FigureObject/"
        }
    else:
        raise ValueError(f"Unknown server_name: {server_name}. Use 'autodl' or 'lambda'")
    
    return paths


def ensure_directories(server_name):
    """
    Ensure all necessary directories exist for the given server configuration.
    
    Args:
        server_name: Server name to get paths for
    """
    paths = get_paths(server_name)
    
    # Extract directory paths from file paths
    directories_to_create = set()
    
    # Add directories from paths that end with /
    for key, path in paths.items():
        if path.endswith('/'):
            directories_to_create.add(path)
        else:
            # For file paths, extract the directory
            dir_path = os.path.dirname(path)
            if dir_path:
                directories_to_create.add(dir_path)
    
    # Create directories
    for directory in directories_to_create:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")


def natural_key(fname: str):
    """
    提取文件名中的数字序列并按整数比较，实现自然排序。
    'image_10.jpg' → [10]
    'frame12_part3.png' → [12, 3]
    """
    return [int(s) if s.isdigit() else s
            for s in re.split(r'(\d+)', fname)]
    
def resize_npz_5ch(
    arr,
    target_hw: tuple[int, int] = (512, 512),
    dtype_out=np.float32,             # or np.float16 if you prefer
    verbose: bool = True
) -> np.ndarray:
    """
    Load a 5-channel 4 K RGB+flow tensor and resize it to `target_hw`.
    
    • Scales U (x-flow) and V (y-flow) by the same factors used for resizing.
    • Uses bilinear interpolation for all channels (adequate for flow too).
    """
    # ---------- 1. load -----------------------------------------------------
    arr = arr.astype(np.float32)                                   # safe math
    
    H0, W0 = arr.shape[:2]
    Ht, Wt = target_hw
    scale_h, scale_w = Ht / H0, Wt / W0          # ≈ 0.237 and 0.133 for 4 K→512²
    
    if verbose:
        # Print flow statistics before scaling
        flow_u = arr[..., 3]
        flow_v = arr[..., 4]
        print(f"  Flow statistics before scaling:")
        print(f"    U: min={flow_u.min():.2f}, max={flow_u.max():.2f}, mean={flow_u.mean():.2f}")
        print(f"    V: min={flow_v.min():.2f}, max={flow_v.max():.2f}, mean={flow_v.mean():.2f}")
    
    # ---------- 2. rescale flow magnitudes ---------------------------------
    arr[..., 3] *= scale_w      # U channel
    arr[..., 4] *= scale_h      # V channel
    
    if verbose:
        print(f"  Flow scaling factors: U *= {scale_w:.3f}, V *= {scale_h:.3f}")
    
    # ---------- 3. spatial resize  -----------------------------------------
    arr = tf.image.resize(arr, target_hw, method="bilinear").numpy()
    
    if verbose:
        # Print flow statistics after scaling
        flow_u_scaled = arr[..., 3]
        flow_v_scaled = arr[..., 4]
        print(f"  Flow statistics after scaling:")
        print(f"    U: min={flow_u_scaled.min():.2f}, max={flow_u_scaled.max():.2f}, mean={flow_u_scaled.mean():.2f}")
        print(f"    V: min={flow_v_scaled.min():.2f}, max={flow_v_scaled.max():.2f}, mean={flow_v_scaled.mean():.2f}")
    
    return arr.astype(dtype_out)

def play_vid(path_vid):
    """
    This function plays the imported vide based on the path.
    :param path_vid: The path of the vide
    :return: None
    """
    cap = cv2.VideoCapture(path_vid)
    while cap.isOpened():
        ret, frame = cap.read()
        color = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        cv2.imshow('frame', color)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def get_fps(path_vid):
    """
    This function return the recorded FPS of the vide.
    :param path_vid: The path of the vide
    :return: The video file's FPS
    """
    cap = cv2.VideoCapture(path_vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    cv2.destroyAllWindows()
    return fps


def vid_to_frame(path_vid, mode):
    """
    Extracting the frames from the video file.
    :param path_vid: The path of the video
    :param mode: Based on the opened file for this project 1) Fire, 2) No-Fire:Lake Mary, 3) Extracting for test set
    :return: None
    """
    vidcap = cv2.VideoCapture(path_vid)
    success, image = vidcap.read()
    count = 0
    while success:
        if mode == 'Fire':
            cv2.imwrite("frames/all/frame%d.jpg" % count, image)  # Save JPG file
        elif mode == 'Lake_Mary':
            cv2.imwrite("frames/lakemary/lake_frame%d.jpg" % count, image)  # Save JPG file
        elif mode == 'Test_Frame':
            # cv2.imwrite("frames/Test_frame/Fire/test_fire_frame%d.jpg" % count, image)  # Save JPG file for FIRE
            cv2.imwrite("frames/Test_frame/No_Fire/test_nofire_frame%d.jpg" % count, image)
            # Save JPG file for NO_FIRE, for NO Fire frames uncomment this line and comment the previous line
        success, image = vidcap.read()
        print('Extract new frame: ', success, ' frame = ', count)
        count += 1


def resize(path_all, path_resize, mode):
    """
    Resizing the imported images to the project and save them on drive based on the dimension parameter.
    :param path_all: The directory of loaded images to the project
    :param path_resize: The directory to save the resized files
    :param mode: Fire, No_Fire(lake mary), or the test data
    :return: None
    """
    image_names_dir = os.listdir(path_all)
    if mode == 'Test_Frame':
        # image_names_dir = os.listdir(path_all + 'Fire')  # This is for the FIRE DIR (Test)
        image_names_dir = os.listdir(path_all + 'No_Fire')  # This is for the No_FIRE DIR (Test)
    image_names_dir.sort()
    new_width = new_size.get('width')
    new_height = new_size.get('height')
    dimension = (new_width, new_height)

    count = 0
    for image in image_names_dir:
        # print(resized_img.shape)
        # cv2.imshow('output', resized_img)
        if mode == 'Fire':
            img = cv2.imread(path_all + '/' + image)
            resized_img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path_resize + '/resized_' + image, resized_img)
        elif mode == 'Lake_Mary':
            img = cv2.imread(path_all + '/' + image)
            resized_img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path_resize + '/lake_resized_' + image, resized_img)
        elif mode == 'Test_Frame':
            # img = cv2.imread(path_all + 'Fire/' + image)
            img = cv2.imread(path_all + 'No_Fire/' + image)
            resized_img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
            # cv2.imwrite(path_resize + 'Fire/resized_' + image, resized_img)  # Resize for Fire (Test Data)
            cv2.imwrite(path_resize + 'No_Fire/resized_' + image, resized_img)  # Resize for NoFire (Test Data)

        print('Image Resized ' + str(count) + ' : resized_' + image)
        count += 1


def rename_all_files(path=None):
    """
    This function returns all the files included in the path directory. This function is used for the fire segmentation
    challenge to have the same name for both the frame and the peered mask.
    :param path: The input directory to rename the included files
    :return: None
    """
    regex = re.compile(r'\d+')
    if path == "Image":
        path_dir = "frames/Segmentation/Data/Images"
    elif path == "Mask":
        path_dir = "frames/Segmentation/Data/Masks"
    else:
        print("Wrong Path for renaming!")
        print("Exit with return")
        return
    files_images = os.listdir(path_dir)
    files_images.sort()
    for count, filename in enumerate(files_images):
        num_ex = regex.findall(filename)
        if path == "Image":
            dst = "image_" + num_ex[0] + ".jpg"
        elif path == "Mask":
            dst = "image_" + str(int(num_ex[0]) - 1) + ".png"
        else:
            print("\nWrong path option ... ")
            return 0
        dst = path_dir + '/' + dst
        src = path_dir + '/' + filename
        os.rename(src, dst)
        print("count = ", count)
