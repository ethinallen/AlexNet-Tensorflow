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

# AlexNet Network Definition
def alex_net(batch_size, patch_size, depth, hidden_size, data):

    # We'll be using this variables frequently
    image_size = data["image_size"]
    num_labels = data["num_labels"]
    num_channels = data["num_channels"]

    # Create the tensorflow graph
    graph = tf.Graph()

    with graph.as_default():
        '''
            as_default() is the method that creates a context manager object corresponding to
            what you got out of "tf.Graph" That context manager pushes new Graph instance to
            thread-local stack so that all new TensorFlow ops are created for that graph.
        '''

        
