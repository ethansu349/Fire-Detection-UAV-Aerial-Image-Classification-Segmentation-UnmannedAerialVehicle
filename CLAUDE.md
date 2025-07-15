# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the Fire Detection UAV component.

## Component Overview

This component implements fire detection using UAV aerial imagery with **primary focus on segmentation (pixel-level fire detection)** using deep learning models. While classification capabilities exist, the main research focus and integration with optical flow occurs through the segmentation pipeline.

## Core Architecture

### Data Flow Pipeline
1. **Video Input** → **Frame Extraction** → **Preprocessing** → **Model Training/Inference** → **Results**
2. **Primary Segmentation Pipeline**: Images + Masks → U-Net → Pixel-level fire masks (MAIN FOCUS)
3. **Secondary Classification Pipeline**: Images → Xception-based CNN → Binary classification (Fire/No-Fire)

### File Structure and Dependencies

**Entry Point**:
- `main.py` - Central dispatcher that routes execution based on config Mode

**Core Implementation Files**:
- `config.py` - Central configuration containing all parameters, paths, and mode settings
- `utils.py` - Utilities including:
  - Video/image processing (frame extraction, resizing)
  - Path management (`get_paths()`, `ensure_directories()`)
  - 5-channel data resizing (`resize_npz_5ch()` with flow scaling)
  - Natural sorting for file operations
- `segmentation.py` - **PRIMARY**: U-Net based fire segmentation training and inference
  - Supports both 3-channel (RGB) and 5-channel (RGB+Flow) inputs
  - Custom `BinaryIoU` metric for proper binary segmentation evaluation
  - Flow normalization functions (`normalize_flow()`, `load_flow_stats()`)
  - Optimized NPZ loading with automatic resizing
- `training.py` - Keras/TensorFlow fire classification model training using Xception architecture  
- `classification.py` - Model inference and evaluation for trained classification models
- `plotdata.py` - Visualization utilities (updated to handle 5-channel data display)

**Data Dependencies**:
- `frames/Training/` - Training data (Fire/, No_Fire/ subdirectories)
- `frames/Test/` - Test data (Fire/, No_Fire/ subdirectories) 
- `frames/Segmentation/Data/` - Segmentation data (Images/, Masks/ subdirectories)
- `video/` - Input video files for frame extraction
- `Output/` - Generated results, models, and visualizations
- **5-Channel Data** (when Merge_mode=True):
  - NPZ files location configured via `dir_merges` in `get_paths()`
  - Default: `/lambda/nfs/fireseg/data/512x512merged`

## Common Development Commands

```bash
# Set working directory
cd Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle

# Train fire classification model
python main.py  # Ensure Mode = 'Training' in config.py

# Run fire classification inference  
python main.py  # Ensure Mode = 'Classification' in config.py

# Train/run fire segmentation
python main.py  # Ensure Mode = 'Segmentation' in config.py

# Extract video frames
python main.py  # Set Mode = 'Fire', 'Lake_Mary', or 'Test_Frame' in config.py

# Rename segmentation files for proper pairing
python main.py  # Set Mode = 'Rename' in config.py
```

## 5-Channel Segmentation Workflow

### 1. Prepare NPZ Data
```bash
cd ../optical_flow_data_process
python concate_flow_jpg.py  # Merges RGB+Flow → NPZ files
```

### 2. Configure for 5-Channel Training
```python
# In config.py:
config_segmentation = {
    "CHANNELS": 5,           # Must be 5 for RGB+Flow
    "Merge_mode": True,      # Enable NPZ loading
    "Test_mode": False,      # Use full dataset
    "flow_norm_mode": "raw", # Or "tanh", "zscore", "minmax"
    # ... other parameters
}
```

### 3. Update Paths (if needed)
```python
# In utils.py, update get_paths() function:
'dir_merges': "/path/to/your/npz/files",
```

### 4. Run Training
```bash
python main.py  # With Mode = 'Segmentation'
```

## Configuration Management

### Mode Control (config.py)
The `Mode` variable controls operation type:
- `'Training'` - Train Xception-based fire classification model
- `'Classification'` - Run inference on test dataset  
- `'Segmentation'` - Train/run U-Net fire segmentation
- `'Fire'` - Extract frames from fire video 
- `'Lake_Mary'` - Extract frames from Lake Mary video
- `'Test_Frame'` - Extract test frames
- `'Rename'` - Rename files for segmentation data pairing

### Key Configuration Parameters

**Image Dimensions**:
- `new_size` - Classification input size (256x256)
- `segmentation_new_size` - Segmentation input size (512x512)

**Training Parameters**:
- `Config_classification` - Classification training config (batch_size=32, epochs=40)
- `config_segmentation` - Segmentation training config (batch_size=16, epochs=30)
  - `Test_mode` - Limits dataset to first 50 samples when True
  - `Merge_mode` - Uses 5-channel RGB+flow data when True
  - `CHANNELS` - 3 for RGB-only, 5 for RGB+Flow (must match Merge_mode)
  - `flow_norm_mode` - Flow normalization: 'raw', 'tanh', 'zscore', 'minmax'
  - `flow_stats_path` - Path to flow statistics file (for zscore/minmax modes)
  - `server_name` - Server configuration ('autodl' or 'lambda')

## Model Architectures

### Classification Model (training.py)
- **Base**: Modified Xception network with reduced complexity
- **Input**: 256x256x3 RGB images  
- **Architecture**: 
  - Single separable conv block (size=8) instead of full Xception
  - Rescaling layer (1.0/255) for normalization
  - Global average pooling + dropout (0.5)
  - Binary output with sigmoid activation
- **Training**: Adam optimizer, binary crossentropy loss, 40 epochs default

### Segmentation Model (segmentation.py) 
- **Base**: U-Net architecture for pixel-level segmentation
- **Input**: 512x512x3 RGB images (or 512x512x5 for RGB+flow mode)
- **Architecture**:
  - Encoder: 4 downsampling blocks (16→32→64→128→256 filters)
  - Decoder: 4 upsampling blocks with skip connections
  - Each block: Conv2D + BatchNorm + ELU + Dropout + MaxPool/Conv2DTranspose
  - Output: Single channel with sigmoid activation
- **Training**: Adam optimizer, binary crossentropy loss, early stopping (patience=5)

## Data Processing Utilities (utils.py)

### Path Management (NEW)
- `get_paths(server_name)` - Centralized path configuration for different servers
  - Supports 'autodl' and 'lambda' server configurations
  - Returns dictionary with all necessary paths
  - Dynamic output paths with server_name prefix
- `ensure_directories(server_name)` - Automatically creates all required directories

### Video Processing
- `play_vid(path)` - Display video file
- `get_fps(path)` - Extract video frame rate
- `vid_to_frame(path, mode)` - Extract all frames from video

### Image Processing  
- `resize(path_all, path_resize, mode)` - Batch resize images using OpenCV
- `natural_key(fname)` - Natural sorting for filenames with numbers
- `resize_npz_5ch(arr, target_hw, dtype_out, verbose)` - Enhanced 5-channel resizing
  - Properly scales optical flow U/V components
  - Verbose mode shows flow statistics before/after scaling
  - Supports different output data types

### File Management
- `rename_all_files(path)` - Rename segmentation images/masks for proper pairing

## Integration with Optical Flow (SEGMENTATION ONLY)

### 5-Channel Data Support in Segmentation Pipeline
**NOTE: Optical flow integration occurs ONLY with the segmentation task, not classification.**

When `Merge_mode = True` in config_segmentation:
- Loads 5-channel .npz files containing RGB+UV flow data
- Uses `resize_npz_5ch()` to properly scale flow components during resize
- U-Net segmentation model automatically adapts input channels based on data
- **Classification pipeline remains RGB-only (3 channels)**

### Flow Data Processing for Segmentation
- Flow U and V components scaled by width/height ratios during resize
- TensorFlow bilinear interpolation preserves flow field integrity
- Integration point with `optical_flow_data_process/` component
- **Only applies to segmentation workflow, not classification**

### Flow Normalization (NEW in segmentation.py)
- `normalize_flow(flow, mode, stats)` - Flexible flow normalization
  - 'raw': Keep original values
  - 'tanh': Normalize using tanh(flow/20.0)
  - 'zscore': Standardize using global statistics
  - 'minmax': Scale to [-1, 1] using global min/max
- `load_flow_stats(stats_path)` - Load global flow statistics from Excel/CSV

## Training Workflows

### Primary: Segmentation Training (MAIN FOCUS)
1. Load images from data directories based on mode
2. **NEW: Automatic resolution checking and resizing for NPZ files**
   - Only checks first file to determine resize parameters
   - Applies same parameters to all files (optimized)
   - Detailed logging of resize operations
3. Apply early stopping to prevent overfitting
4. **Support both RGB-only (3-channel) and RGB+flow (5-channel) inputs**
5. **Primary integration point with optical flow data**
6. Generate segmentation test results and visualizations
7. **NEW: Custom BinaryIoU metric for accurate binary segmentation evaluation**

### Secondary: Classification Training
1. Load training data from `frames/Training/Fire/` and `frames/Training/No_Fire/`
2. Apply data augmentation (horizontal flip, rotation)
3. Calculate class weights for balanced training
4. Train modified Xception model with validation split (20%)
5. Save best model and generate training plots
6. **NOTE: No optical flow integration - RGB only**

## Output and Visualization

### Generated Outputs
- **Models**: Saved in `Output/Models/` and `/lambda/nfs/fireseg/seg_output/model_checkpoints/`
- **Figures**: Training plots, confusion matrices in `Output/Figures/` 
- **Figure Objects**: Pickled matplotlib objects in `Output/FigureObject/`

### Visualization Functions (plotdata.py)
- `plot_training()` - Training loss/accuracy curves
- `plot_confusion_matrix()` - Classification confusion matrix
- `plot_segmentation_test()` - Segmentation results comparison
  - **Updated**: Now handles 5-channel data by displaying only RGB channels
  - Uses matplotlib.pyplot.imshow instead of deprecated skimage.imshow
- `plot_metrics()` - Comprehensive training metrics

## Dependencies and Requirements

- Python 3.6+
- TensorFlow 2.3.0+ 
- Keras 2.4.0+
- OpenCV (cv2)
- NumPy, SciPy, matplotlib
- scikit-image, tqdm

## Dataset Requirements

### FLAME Dataset from IEEE Dataport
- **Items 7 & 8**: Classification data for `frames/Training/` and `frames/Test/`
- **Items 9 & 10**: Segmentation data for `frames/Segmentation/Data/`

### Expected Directory Structure
```
frames/
├── Training/
│   ├── Fire/*.jpg
│   └── No_Fire/*.jpg  
├── Test/
│   ├── Fire/*.jpg
│   └── No_Fire/*.jpg
└── Segmentation/Data/
    ├── Images/*.jpg
    └── Masks/*.png
```

## Recent Optimizations

### NPZ Data Loading (segmentation.py)
- **One-time resolution check**: Only first NPZ file checked for dimensions
- **Batch parameter application**: Same resize parameters applied to all files
- **Reduced logging**: Detailed stats only for first file, summary at end
- **Performance improvement**: Eliminates N-1 redundant checks

### RGB+Flow Concatenation (via optical_flow_data_process/concate_flow_jpg.py)
- **Automatic RGB resizing**: 4K images automatically resized to match 512×512 flow
- **One-time size detection**: Similar optimization as NPZ loading
- **Cleaner output**: Only errors printed during processing

### Path Management System
- **Centralized configuration**: All paths managed through `get_paths()`
- **Server flexibility**: Easy switching between 'autodl' and 'lambda' servers
- **Automatic directory creation**: `ensure_directories()` prevents missing directory errors
- **Dynamic output paths**: Server name included in output directories

## Integration Points

### Data Input Sources
- Video files processed into frame sequences
- **5-channel RGB+flow data from optical flow processing component (SEGMENTATION ONLY)**
- FLAME dataset images and ground truth masks
- **Classification uses RGB-only (3-channel) data**

### Data Output Targets  
- **Primary: Trained segmentation models with optical flow enhancement**
- **Primary: Pixel-level fire segmentation masks**
- Secondary: Trained classification models (RGB-only)
- Secondary: Fire detection results and confidence scores
- Training metrics and visualizations

## Troubleshooting

### Common Issues
- **Early stopping at epoch 7**: Early stopping patience=5 triggers when validation loss stops improving
- **Image display problems**: Check for double normalization in data preprocessing
- **Path errors**: Ensure data directories match expected structure
- **Memory issues**: Reduce batch size or enable Test_mode for limited dataset
- **5-channel image display error** (NEW): 
  - Error: `Invalid shape (512, 512, 5) for image data`
  - Solution: Fixed in plotdata.py to display only RGB channels `[..., :3]`
- **Double normalization issue** (FIXED):
  - Previous: Lambda layer + manual division caused values near 0
  - Solution: Removed Lambda layer, data types changed to float32
- **IoU metric stuck at ~0.497** (FIXED):
  - Cause: MeanIoU expects discrete classes but receives probabilities
  - Solution: Added custom BinaryIoU metric with thresholding

### Configuration Tips
- Set `Test_mode = True` for quick testing with limited data
- Use `Merge_mode = True` when integrating with optical flow data
- **IMPORTANT**: Set `CHANNELS = 5` when `Merge_mode = True`
- Adjust early stopping patience based on training stability
- Verify image/mask filename correspondence for segmentation
- Try different `flow_norm_mode` values if flow data causes issues:
  - 'raw': Use original flow values (default)
  - 'tanh': Good for normalizing large flow ranges
  - 'zscore' or 'minmax': Requires flow statistics file