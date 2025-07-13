"""
#################################
 Fire Segmentation on Fire Class to extract fire pixels from each frame based on the Ground Truth data (masks)
 Train, Validation, Test Data: Items (9) and (10) on https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs
 Keras version: 2.4.0
 Tensorflow Version: 2.3.0
 GPU: Nvidia RTX 2080 Ti
 OS: Ubuntu 18.04
################################
"""

#########################################################
# import libraries
import os
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

from config import config_segmentation, segmentation_new_size, server_name
from plotdata import plot_segmentation_test
from utils import natural_key, resize_npz_5ch, get_paths, ensure_directories

#########################################################
# Global parameters and definition

class BinaryIoU(tf.keras.metrics.Metric):
    """Custom Binary IoU metric that properly handles probability outputs."""
    
    def __init__(self, threshold=0.5, name='binary_iou', **kwargs):
        super(BinaryIoU, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the state variables."""
        # Threshold predictions to get binary values
        y_pred_binary = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate TP, FP, FN
        true_positives = tf.reduce_sum(y_true * y_pred_binary)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred_binary)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred_binary))
        
        # Update state
        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)
    
    def result(self):
        """Computes and returns the IoU metric."""
        denominator = self.true_positives + self.false_positives + self.false_negatives
        # Avoid division by zero
        denominator = tf.maximum(denominator, tf.keras.backend.epsilon())
        return self.true_positives / denominator
    
    def reset_state(self):
        """Resets all the state variables."""
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)


METRICS = [
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.Accuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.MeanIoU(num_classes=2, name='iou'),
    BinaryIoU(name='binary_iou'),  # Custom binary IoU with thresholding
    tf.keras.metrics.BinaryAccuracy(name='bin_accuracy'),
]


#########################################################
# Function definition

def normalize_flow(flow, mode='raw', stats=None):
    """
    Normalize optical flow data using different strategies.
    
    Args:
        flow: Flow data with shape (..., 2) where last dimension is (u, v)
        mode: Normalization mode - 'raw', 'tanh', 'zscore', 'minmax'
        stats: Dictionary containing global statistics (mean, std, min, max) for zscore/minmax modes
    
    Returns:
        Normalized flow data
    """
    if mode == 'raw':
        # Keep original flow values
        return flow
    
    elif mode == 'tanh':
        # Normalize using tanh to roughly [-1, 1] range
        # 20.0 is a typical flow magnitude for moderate motion
        return np.tanh(flow / 20.0)
    
    elif mode == 'zscore':
        # Standardize using global statistics
        if stats is None:
            raise ValueError("Stats required for zscore normalization")
        return (flow - stats['mean']) / (stats['std'] + 1e-8)
    
    elif mode == 'minmax':
        # Normalize to [-1, 1] range using global min/max
        if stats is None:
            raise ValueError("Stats required for minmax normalization")
        # Clip to handle outliers
        flow_clipped = np.clip(flow, stats['min'], stats['max'])
        return 2.0 * (flow_clipped - stats['min']) / (stats['max'] - stats['min'] + 1e-8) - 1.0
    
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def load_flow_stats(stats_path):
    """
    Load flow statistics from a file (e.g., output from flow_stats.py).
    
    Args:
        stats_path: Path to statistics file (Excel or CSV)
    
    Returns:
        Dictionary with flow statistics for normalization
    """
    if stats_path is None:
        return None
    
    try:
        import pandas as pd
        
        # Try to load as Excel first, then CSV
        try:
            df = pd.read_excel(stats_path)
        except:
            df = pd.read_csv(stats_path)
        
        # Extract statistics for U and V channels
        # Assuming the stats file has columns for u and v statistics
        stats = {
            'mean': np.array([df.loc[df['Unnamed: 0'] == 'u', 'mean'].values[0],
                            df.loc[df['Unnamed: 0'] == 'v', 'mean'].values[0]]),
            'std': np.array([df.loc[df['Unnamed: 0'] == 'u', 'std'].values[0],
                           df.loc[df['Unnamed: 0'] == 'v', 'std'].values[0]]),
            'min': np.array([df.loc[df['Unnamed: 0'] == 'u', 'min'].values[0],
                           df.loc[df['Unnamed: 0'] == 'v', 'min'].values[0]]),
            'max': np.array([df.loc[df['Unnamed: 0'] == 'u', 'max'].values[0],
                           df.loc[df['Unnamed: 0'] == 'v', 'max'].values[0]])
        }
        
        print(f"Loaded flow statistics from {stats_path}")
        print(f"Flow U - mean: {stats['mean'][0]:.3f}, std: {stats['std'][0]:.3f}, range: [{stats['min'][0]:.3f}, {stats['max'][0]:.3f}]")
        print(f"Flow V - mean: {stats['mean'][1]:.3f}, std: {stats['std'][1]:.3f}, range: [{stats['min'][1]:.3f}, {stats['max'][1]:.3f}]")
        
        return stats
    
    except Exception as e:
        print(f"Warning: Could not load flow statistics from {stats_path}: {e}")
        return None


def segmentation_keras_load():
    """
    This function trains a DNN model for the fire segmentation based on the U-NET Structure.
    Arxiv Link for U-Net: https://arxiv.org/abs/1505.04597
    :return: None, Save the model and plot the predicted fire masks on the validation dataset.
    """

    """ Defining general parameters """
    batch_size = config_segmentation.get('batch_size')
    img_size = (segmentation_new_size.get("width"), segmentation_new_size.get("height"))
    img_width = img_size[0]
    img_height = img_size[1]
    epochs = config_segmentation.get('Epochs')
    img_channels = config_segmentation.get('CHANNELS')
    num_classes = config_segmentation.get("num_class")
    merge_mode = config_segmentation.get("Merge_mode")
    test_mode = config_segmentation.get("Test_mode")
    
    # Load flow statistics if using merge mode and stats path is provided
    if merge_mode and config_segmentation.get('flow_stats_path'):
        config_segmentation['flow_stats'] = load_flow_stats(config_segmentation['flow_stats_path'])

    # Get paths based on server configuration
    paths = get_paths(server_name)
    
    # Ensure all output directories exist
    ensure_directories(server_name)
    
    """ Defining the directory of the images and masks """
    dir_images = paths['dir_images']
    dir_masks = paths['dir_masks']
    dir_merges = paths['dir_merges']
    
    """ Defining the model figure file directory / path"""
    model_fig_file = paths['model_fig_file']

    

    """ Start reading data (Frames and masks) and save them in Numpy array for Training, Validation and Test"""
    if merge_mode:
        print("merge on, input file image now is merged 5ch")
        allfiles_image = [fname for fname in tqdm(os.listdir(dir_merges)) if fname.endswith(".npz") and not fname.startswith(".")]
        allfiles_image.sort(key=natural_key)
        allfiles_image = [os.path.join(dir_merges, fname) for fname in allfiles_image]
    else:
        allfiles_image = [fname for fname in tqdm(os.listdir(dir_images)) if fname.endswith(".jpg") and not fname.startswith(".")]
        allfiles_image.sort(key=natural_key)
        allfiles_image = [os.path.join(dir_images, fname) for fname in allfiles_image]
    
    allfiles_mask = [fname for fname in tqdm(os.listdir(dir_masks)) if fname.endswith(".png") and not fname.startswith(".")]
    allfiles_mask.sort(key=natural_key)
    allfiles_mask = [os.path.join(dir_masks, fname) for fname in allfiles_mask]
    print("X number | Mask number: ")
    print(len(allfiles_image), len(allfiles_mask))
    if merge_mode:
        # since flo files always have one file less than original images files, the last frame doesnot have flow result.
        print("merge on, input mask now is one file less than original")
        allfiles_mask = allfiles_mask[:-1]
    print("After merge: X number | Mask number: ")
    print(len(allfiles_image), len(allfiles_mask)) 
    
    print("Number of samples:", len(allfiles_image))
    print("Input_path", "|", "Target_path: ")
    for input_path, target_path in tqdm(zip(allfiles_image[:10], allfiles_mask[:10])):
        print(input_path, "|", target_path)

    if test_mode:
        allfiles_image = allfiles_image[:50]
        allfiles_mask = allfiles_mask[:50]
        
    total_samples = len(allfiles_mask)
    train_ratio = config_segmentation.get("train_set_ratio")
    val_samples = int(total_samples * (1 - train_ratio))
    random.Random(1337).shuffle(allfiles_image)
    random.Random(1337).shuffle(allfiles_mask)
    
    train_img_paths = allfiles_image[:-val_samples]
    train_mask_paths = allfiles_mask[:-val_samples]
    val_img_paths = allfiles_image[-val_samples:]
    val_mask_paths = allfiles_mask[-val_samples:]

    # create nan train and val set variables
    x_train = np.zeros((len(train_img_paths), img_height, img_width, img_channels), dtype=np.float32)
    y_train = np.zeros((len(train_mask_paths), img_height, img_width, 1), dtype=np.bool_)
    x_val = np.zeros((len(val_img_paths), img_height, img_width, img_channels), dtype=np.float32)
    y_val = np.zeros((len(val_mask_paths), img_height, img_width, 1), dtype=np.bool_)
    
    print('\nLoading training images: ', len(train_img_paths), 'images ...')
    for n, file_ in tqdm(enumerate(train_img_paths)):
        if merge_mode:
            # print("merge on, x train resize to 512x512")
            rgbuv = np.load(file_)["rgbuv"]
            # RGB channels are already in [0, 1] range from concate_flow_jpg.py
            # Flow channels need normalization based on config
            rgb = rgbuv[..., :3]  # Already in [0, 1]
            flow = rgbuv[..., 3:]  # Raw flow values
            
            # Get flow normalization mode from config (default to 'raw')
            flow_norm_mode = config_segmentation.get('flow_norm_mode', 'raw')
            flow_stats = config_segmentation.get('flow_stats', None)
            
            # Normalize flow channels
            flow_normalized = normalize_flow(flow, mode=flow_norm_mode, stats=flow_stats)
            
            # Combine normalized data
            x_train[n] = np.concatenate([rgb, flow_normalized], axis=-1)
        else:
            # Regular mode: load JPG images which are in [0, 255] range
            img = tf.keras.preprocessing.image.load_img(file_, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            # Normalize to [0, 1] for consistency with merge mode
            x_train[n] = img_array / 255.0

    print('\nLoading training masks: ', len(train_mask_paths), 'masks ...')
    for n, file_ in tqdm(enumerate(train_mask_paths)):
        img = tf.keras.preprocessing.image.load_img(file_, target_size=img_size, color_mode="grayscale")
        y_train[n] = np.expand_dims(img, axis=2)
        # y_train[n] = y_train[n] // 255

    print('\nLoading test images: ', len(val_img_paths), 'images ...')
    for n, file_ in tqdm(enumerate(val_img_paths)):
        if merge_mode:
            # print("merge on, x val resize to 512x512")
            rgbuv = np.load(file_)["rgbuv"]
            # RGB channels are already in [0, 1] range from concate_flow_jpg.py
            # Flow channels need normalization based on config
            rgb = rgbuv[..., :3]  # Already in [0, 1]
            flow = rgbuv[..., 3:]  # Raw flow values
            
            # Get flow normalization mode from config (default to 'raw')
            flow_norm_mode = config_segmentation.get('flow_norm_mode', 'raw')
            flow_stats = config_segmentation.get('flow_stats', None)
            
            # Normalize flow channels
            flow_normalized = normalize_flow(flow, mode=flow_norm_mode, stats=flow_stats)
            
            # Combine normalized data
            x_val[n] = np.concatenate([rgb, flow_normalized], axis=-1)
        else:
            # Regular mode: load JPG images which are in [0, 255] range
            img = tf.keras.preprocessing.image.load_img(file_, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            # Normalize to [0, 1] for consistency with merge mode
            x_val[n] = img_array / 255.0

    print('\nLoading test masks: ', len(val_mask_paths), 'masks ...')
    for n, file_ in tqdm(enumerate(val_mask_paths)):
        img = tf.keras.preprocessing.image.load_img(file_, target_size=img_size, color_mode="grayscale")
        y_val[n] = np.expand_dims(img, axis=-1)
        # y_val[n] = y_val[n] // 255

    print("----------X_train[0]------------")
    print(x_train[0])
    print("--------------------------------")
    print("----------Y_train[0]------------")
    print(y_train[0])
    print("--------------------------------")

    """ Plot some random data: frame and mask (gTruth)"""
    idx_rand = random.randint(0, len(train_img_paths))
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x_train[idx_rand][...,:3])
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(y_train[idx_rand]))
    plt.axis('off')
    plt.show()

    tf.keras.backend.clear_session()

    """ Training the Model ... """
    model = model_unet_kaggle(img_height, img_width, img_channels, num_classes)
    tf.keras.utils.plot_model(model, to_file=model_fig_file, show_shapes=True)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)
    # change h5 file storage path.
    # change h5 file storage path based on server configuration
    checkpoint = tf.keras.callbacks.ModelCheckpoint(paths['checkpoint'], save_best_only=True)

    
    early_stopper = tf.keras.callbacks.EarlyStopping(patience=5)

    results = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopper, checkpoint])

    """ Prediciting mask using the model ... """
    # model_predict = tf.keras.models.load_model("FireSegmentation_fifth.h5")
    # Load model from the configured checkpoint path
    model_predict = tf.keras.models.load_model(paths['checkpoint'])
    

    preds_val = model.predict(x_val, verbose=1)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)

    """ Plotting a few generated masks from the model and compare them with the Ground Truth Masks ... """
    plot_segmentation_test(xval=x_val, yval=y_val, ypred=preds_val_t, num_samples=6)

def model_unet_kaggle(img_hieght, img_width, img_channel, num_classes):
    """
    This function returns a U-Net Model for this binary fire segmentation images:
    Arxiv Link for U-Net: https://arxiv.org/abs/1505.04597
    :param img_hieght: Image Height
    :param img_width: Image Width
    :param img_channel: Number of channels in each image
    :param num_classes: Number of classes based on the Ground Truth Masks
    :return: A convolutional NN based on Tensorflow and Keras
    """
    inputs = Input((img_hieght, img_width, img_channel))

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
