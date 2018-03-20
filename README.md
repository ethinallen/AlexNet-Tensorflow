# AlexNet-Tensorflow

An Implementation of AlexNet Convolutional Neural Network Architecture by Krizhevsky, Sutskever & Hinton using Tensorflow.

---

This is a simple implementation of the great paper **ImageNet Classification with Deep Convolutional Neural Networks** by **Alex Krizhevsky, Ilya Sutskever** and **Geoffrey Hinton**. Some key takeaways of this paper is stated here in addition to the implemented model in the *alexnet.py* model.

---

# Key Takeaways

## Dataset: ImageNet

The dataset used for this project is a subset of the main **ImageNet** dataset. This subset includes **1.2M** high-resolution images in **1000** categories. The main ImageNet dataset contains more than **15M** high-res images in about **22000**categories. **1.2M** images is a massive number of images which urges a model with **large learning capacity**.

## Capacity of CNNs

The capacity of CNNs is determined with their **Breadth** and **Depth**. Note that if the network size is large, then the low number of training samples can cause **overfitting**. It is also important to mention that most of the parameters of the model is layed in the **fully-connected** layers. In this specific example(this paper), each of the convolutional layers contains no more than **1%** of the model's parameters.




