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
FC_HIDDEN_SIZE = 4096

# Global dataset dictionary
dataset_dict = {
    "image_size": 224,
    "num_channels": 3,
    "num_labels": 22000,
    "filter_shape": [5, 5],
    "pool_shape": [2, 2]
}

dataset_dict["total_image_size"] = dataset_dict["image_size"] * dataset_dict["image_size"]

# Define the input and output placeholders
train = tf.placeholder(tf.float32, shape=[BATCH_SIZE, dataset_dict["image_size"], dataset_dict["image_size"], dataset_dict["num_channels"]])
labels = tf.placeholder(tf.float32, shape=[None, dataset_dict["num_labels"]])

def create_conv_layer(input_shape, num_channels, num_filters, filter_shape, pool_shape, name):
    '''
        Generates a new convolutional layer with the given parameters.
    '''

    # Define the filter window shape
    filter_window_shape = [filter_shape[0], filter_shape[1], num_channels, num_filters]

    # Set initial weights and biases for the filter
    filter_weights = tf.Variable(tf.truncated_normal(filter_window_shape), name=name + '_W')
    filter_biases = tf.Variable(tf.truncated_normal(num_filters), name=name + '_b')

    # Create the main convoloutional layer
    conv_layer = tf.nn.conv2d(input_shape, filter=filter_weights, strides=[1, 1, 1, 1], padding='SAME', name='CONV_1')

    # Add biases to the filter weights
    conv_layer += filter_biases

    # ReLU non-linear activation
    conv_layer = tf.nn.relu(conv_layer, name='ReLU_1')

    # Max pooling layer
    ksize = [1, pool_shape[0], pool_shape[1], 1]       # Pooling window size | first & last args are always set to 1
    strides = [1, 2, 2, 1]
    conv_layer = tf.nn.max_pool(value=input_shape, ksize=ksize, strides=strides, padding='SAME', name='MAX_POOL_1')

    # Return the sub-graph
    return conv_layer


# Generate 5 convolutional layers
c_layer_1 = create_conv_layer(x_shaped, 3, 32, dataset_dict["filter_shape"], dataset_dict["pool_shape"], name='c_layer_1')
c_layer_2 = create_conv_layer(c_layer_1, 3, 64, dataset_dict["filter_shape"], dataset_dict["pool_shape"], name='c_layer_2')
c_layer_3 = create_conv_layer(c_layer_2, 3, 128, dataset_dict["filter_shape"], dataset_dict["pool_shape"], name='c_layer_3')
c_layer_4 = create_conv_layer(c_layer_3, 3, 256, dataset_dict["filter_shape"], dataset_dict["pool_shape"], name='c_layer_4')
c_layer_5 = create_conv_layer(c_layer_4, 3, 512, dataset_dict["filter_shape"], dataset_dict["pool_shape"], name='c_layer_5')
