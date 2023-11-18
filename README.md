# Realtime-Face-Emotion-Recognition

## Overview
This project is a machine learning application that utilizes a convolutional neural network (CNN) to recognize and classify facial emotions in real time. It is built using TensorFlow and OpenCV and employs transfer learning with the pre-trained MobileNetV2 model.

## Prerequisites
The project requires the following Python libraries:
- numpy
- pandas
- matplotlib
- tensorflow
- opencv-python
- opencv-contrib-python

You can install these with the following commands:
pip install numpy pandas matplotlib tensorflow opencv-python opencv-contrib-python

## Dataset
The training data should be organized in folders named '0' to '6', each corresponding to a different emotion:

0: Angry
1: Disgust
2: Fear
3: Happy
4: Neutral
5: Sad
6: Surprise

## Model Architecture
The architecture of the model is based on MobileNetV2, which is a popular convolutional neural network (CNN) designed by Google researchers. MobileNetV2 is optimized for performance on mobile and edge devices, making it suitable for applications that require real-time processing with limited computational resources.

Key aspects of MobileNetV2 include:

1) Depthwise Separable Convolutions: This technique reduces the number of parameters and computational cost compared to standard convolutions. It separates the convolution into a depthwise pass, which applies a single filter per input channel, and a pointwise pass that uses a 1x1 convolution to combine the outputs of the depthwise pass.

2) Inverted Residuals and Linear Bottlenecks: These structures are used within the network to maintain a compact and efficient model. They involve shortcut connections similar to those used in residual networks, which help in the flow of gradients during training, making it easier to train deeper networks.

For the face emotion recognition task, the pre-trained MobileNetV2 model is used as a starting point. The last few layers are customized to fit the specific requirements of the task:

1) Dense Layers: Additional dense (fully connected) layers are added after the base MobileNetV2 model. These layers are responsible for learning higher-level features specific to the emotion classification task.

2) Softmax Output Layer: The final layer is a softmax layer with seven units, corresponding to the seven emotion classes. The softmax function outputs a probability distribution over the classes, with the sum of probabilities equal to one.

## Training
The training process involves several steps:

1) Reading Images: The script loads images from the dataset, which are categorized into different emotion classes.

2) Preprocessing: Each image is resized to 224x224 pixels, which is the standard input size for MobileNetV2. The pixel values are then normalized to the range [0, 1] by dividing by 255. This normalization aids in the training process by scaling the input features to a uniform range.

3) Model Fitting: The preprocessed images and their corresponding labels (emotions) are fed into the model for training. The model learns to associate the image features with the correct emotion labels.

## Evaluation
After training, the model's performance is evaluated on a separate test dataset that has been preprocessed in the same manner as the training data. The accuracy metric is used to measure the model's ability to correctly classify emotions.

## Usage
Once the model is trained and evaluated, it can be used for real-time emotion recognition:

1) Loading Weights: The trained weights are loaded into the model architecture.

2) Running Inference: The model can then take new images as input, preprocess them in the same way as the training images, and output a prediction for each image. The prediction is the model's estimate of the emotion expressed in the image.

## Output
The output of the training process includes metrics such as loss and accuracy:

1) Loss: This is a numerical value that represents the error between the predicted emotion and the actual emotion. The goal of training is to minimize this value.

2) Accuracy: This is the primary metric for evaluating the model's performance. It is calculated as the number of correct predictions divided by the total number of predictions made. The reported accuracy of 96.01% after 40 epochs indicates that the model correctly predicted the emotion 96.01% of the time on the training dataset.
