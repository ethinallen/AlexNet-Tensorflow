# ========================================
# [] File Name : alexnet.py
#
# [] Creation Date : March 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
"""
    Implementation of the AlexNet Convolutional Neural Network Architecture
    Using Tensorflow.
"""
import tensorflow as tf
# General parameters of the model
BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
DROPOUT_KEEP_PROB = 0.5

# Global dataset dictionary
dataset_dict = {
    "image_size": 256,
    "num_channels": 3,
    "num_labels": 22000,
}

dataset_dict["total_image_size"] = dataset_dict["image_size"] * dataset_dict["image_size"]
