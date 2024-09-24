# Cat-Dog Classification Project 🐱🐶

This project is a deep learning model designed to classify images of cats and dogs. Using Convolutional Neural Networks (CNN), the model is trained on a dataset of labeled images to accurately predict whether an image contains a cat or a dog.


## Project Overview
The objective of this project is to build a machine learning model capable of classifying images of cats and dogs. We used a CNN to extract features from the images and train the model to make predictions based on these features.

This project involves:
- Preprocessing of images
- Building and training a CNN model
- Evaluating model performance
- Visualizing predictions


## Model Architecture
We built a Convolutional Neural Network (CNN) with the following architecture:
- **Input Layer**: 128x128 RGB image
- **Convolutional Layers**:  layers with ReLU activation and max-pooling
- **Flattening Layer**
- **Fully Connected Layers**:  dense layers with ReLU activation
- **Output Layer**: Softmax activation with 2 output classes (cat, dog)
