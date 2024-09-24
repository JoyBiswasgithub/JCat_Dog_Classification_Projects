# Cat-Dog Classification Project 🐱🐶

This project is a deep learning model designed to classify images of cats and dogs. Using Convolutional Neural Networks (CNN), the model is trained on a dataset of labeled images to accurately predict whether an image contains a cat or a dog.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The objective of this project is to build a machine learning model capable of classifying images of cats and dogs. We used a CNN to extract features from the images and train the model to make predictions based on these features.

This project involves:
- Preprocessing of images
- Building and training a CNN model
- Evaluating model performance
- Visualizing predictions

## Dataset
The dataset used for this project consists of labeled images of cats and dogs. The dataset can be found on [Kaggle's Cat vs Dog Dataset](https://www.kaggle.com/c/dogs-vs-cats/data).

- Training Set: 25,000 labeled images
- Testing Set: 12,500 labeled images

The dataset is preprocessed to resize images and normalize pixel values.

## Model Architecture
We built a Convolutional Neural Network (CNN) with the following architecture:
- **Input Layer**: 128x128 RGB images
- **Convolutional Layers**: 2 layers with ReLU activation and max-pooling
- **Flattening Layer**
- **Fully Connected Layers**: 2 dense layers with ReLU activation
- **Output Layer**: Softmax activation with 2 output classes (cat, dog)

## Installation
To run this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/cat-dog-classification.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To train the model on the dataset:
1. Download the dataset and extract it to the `data/` directory.
2. Run the training script:
    ```bash
    python train.py
    ```
3. Evaluate the model:
    ```bash
    python evaluate.py
    ```
4. To predict on new images:
    ```bash
    python predict.py --image_path path_to_your_image.jpg
    ```

## Results
- **Training Accuracy**: 98%
- **Validation Accuracy**: 97%
- **Loss**: 0.05

Visualizations of the model’s performance and example predictions can be found in the `notebooks/` directory.

## Contributing
If you wish to contribute to this project, feel free to submit a pull request or open an issue. Contributions are welcome!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
