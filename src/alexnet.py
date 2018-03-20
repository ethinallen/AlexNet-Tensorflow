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

    Paper: ImageNet Classification with Deep Convolutional Neural Networks

    Authors: Krizhevsky, Alex - Sutskever, Ilya - Hinton, Geoffrey E
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
    "num_labels": 1000,
    "filter_shape": [5, 5],
    "pool_shape": [2, 2]
}

# Filter shapes for each layer 
conv_filter_shapes = {
    "c1_filter": [11, 11, 3, 96],
    "c2_filter": [5, 5, 48, 256],
    "c3_filter": [3, 3, 256, 384],
    "c4_filter": [3, 3, 192, 384],
    "c5_filter": [3, 3, 192, 256]
}

# Fully connected shapes
fc_connection_shapes = {
    "f1_shape": [13*13*256, 4096],
    "f2_shape": [4096, 4096],
    "f3_shape": [4096, dataset_dict["num_labels"]]
}

# Weights for each layer
conv_weights = {
    "c1_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c1_filter"]), name="c1_weights"),
    "c2_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c2_filter"]), name="c2_weights"),
    "c3_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c3_filter"]), name="c3_weights"),
    "c4_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c4_filter"]), name="c4_weights"),
    "c5_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c5_filter"]), name="c5_weights"),
    "f1_weights": tf.Variable(tf.truncated_normal(fc_connection_shapes["f1_shape"]), name="f1_weights"),
    "f2_weights": tf.Variable(tf.truncated_normal(fc_connection_shapes["f2_shape"]), name="f2_weights"),
    "f3_weights": tf.Variable(tf.truncated_normal(fc_connection_shapes["f3_shape"]), name="f3_weights")
}

# Biases for each layer
conv_biases = {
    "c1_biases": tf.Variable(tf.truncated_normal(conv_filter_shapes["c1_filter"][3]), name="c1_biases"),
    "c2_biases": tf.Variable(tf.truncated_normal(conv_filter_shapes["c2_filter"][3]), name="c2_biases"), 
    "c3_biases": tf.Variable(tf.truncated_normal(conv_filter_shapes["c3_filter"][3]), name="c3_biases"),
    "c4_biases": tf.Variable(tf.truncated_normal(conv_filter_shapes["c4_filter"][3]), name="c4_biases"),
    "c5_biases": tf.Variable(tf.truncated_normal(conv_filter_shapes["c5_filter"][3]), name="c5_biases"),
    "f1_biases": tf.Variable(tf.truncated_normal(fc_connection_shapes["f1_shape"][3]), name="f1_biases"),
    "f2_biases": tf.Variable(tf.truncated_normal(fc_connection_shapes["f2_shape"][3]), name="f2_biases"),
    "f3_biases": tf.Variable(tf.truncated_normal(fc_connection_shapes["f3_shape"][3]), name="f3_biases")
}

dataset_dict["total_image_size"] = dataset_dict["image_size"] * dataset_dict["image_size"]

# Declare the input and output placeholders
input_img = tf.placeholder(tf.float32, shape=[BATCH_SIZE, dataset_dict["image_size"], dataset_dict["image_size"], dataset_dict["num_channels"]])
img_4d_shaped = tf.reshape(input_img, [-1, dataset_dict["image_size"], dataset_dict["image_size"], dataset_dict["num_channels"]])
labels = tf.placeholder(tf.float32, shape=[None, dataset_dict["num_labels"]])

# Convolution Layer 1 | Response Normalization | Max Pooling | ReLU
c_layer_1 = tf.nn.conv2d(img_4d_shaped, conv_weights["c1_weights"], strides=[1, 4, 4, 1], padding="SAME", name="c_layer_1")
c_layer_1 += conv_biases["c1_biases"]
c_layer_1 = tf.nn.relu(c_layer_1)
c_layer_1 = tf.nn.lrn(c_layer_1, depth_radius=4.5, bias=1.8, alpha=1e-4, beta=0.75)
c_layer_1 = tf.nn.max_pool(c_layer_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

# Convolution Layer 2 | Response Normalization | Max Pooling | ReLU
c_layer_2 = tf.nn.conv2d(c_layer_1, conv_weights["c2_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_2")
c_layer_2 += conv_biases["c2_biases"]
c_layer_2 = tf.nn.relu(c_layer_2)
c_layer_2 = tf.nn.lrn(c_layer_2, depth_radius=4.5, bias=1.8, alpha=1e-4, beta=0.75)
c_layer_2 = tf.nn.max_pool(c_layer_2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="VALID")

# Convolution Layer 3 | ReLU
c_layer_3 = tf.nn.conv2d(c_layer_2, conv_weights["c3_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_3")
c_layer_3 += conv_biases["c3_biases"]
c_layer_3 = tf.nn.relu(c_layer_3)

# Convolution Layer 4 | ReLU
c_layer_4 = tf.nn.conv2d(c_layer_3, conv_weights["c4_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_4")
c_layer_4 += conv_biases["c4_biases"]
c_layer_4 = tf.nn.relu(c_layer_4)

# Convolution Layer 5 | ReLU | Max Pooling
c_layer_5 = tf.nn.conv2d(c_layer_4, conv_weights["c5_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_5")
c_layer_5 += conv_biases["c5_biases"]
c_layer_5 = tf.nn.relu(c_layer_5)
c_layer_5 = tf.nn.max_pool(c_layer_5, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="VALID")

# Fully Connected Layer 1 | Dropout


# Fully Connected Layer 2 | Dropout


# Fully Connected Layer 3 | Softmax


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
c_layer_1 = create_conv_layer(train_4d_shaped, 3, 32, dataset_dict["filter_shape"], dataset_dict["pool_shape"], name='c_layer_1')
c_layer_2 = create_conv_layer(c_layer_1, 32, 64, dataset_dict["filter_shape"], dataset_dict["pool_shape"], name='c_layer_2')
c_layer_3 = create_conv_layer(c_layer_2, 64, 128, dataset_dict["filter_shape"], dataset_dict["pool_shape"], name='c_layer_3')
c_layer_4 = create_conv_layer(c_layer_3, 128, 256, dataset_dict["filter_shape"], dataset_dict["pool_shape"], name='c_layer_4')
c_layer_5 = create_conv_layer(c_layer_4, 256, 512, dataset_dict["filter_shape"], dataset_dict["pool_shape"], name='c_layer_5')

# CONV_LAYER_1 | 