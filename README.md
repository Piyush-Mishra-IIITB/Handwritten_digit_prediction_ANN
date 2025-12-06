# Handwritten Digit Recognition Using ANN

## Project Overview
This project implements an Artificial Neural Network (ANN) to recognize handwritten digits (0–9) from images. The model is trained on the **MNIST dataset**, which contains 60,000 training images and 10,000 testing images of handwritten digits. The project demonstrates the power of deep learning in image classification tasks.

## Features
- Recognizes handwritten digits (0–9) with high accuracy.
- Uses a fully connected Artificial Neural Network.
- Provides visualization of training progress (loss & accuracy).
- Can be extended to other image recognition tasks.

## Dataset
The project uses the **MNIST dataset**, a standard benchmark dataset in the field of machine learning and computer vision. Each image is a 28x28 pixel grayscale image of a single handwritten digit.

## Model Architecture
The ANN model consists of:
- Input layer: 784 neurons (28x28 flattened images)
- Hidden layers: 2 layers with ReLU activation
- Output layer: 10 neurons with Softmax activation (for 10 classes)
- Optimizer: Adam
- Loss function: Categorical Cross-Entropy

