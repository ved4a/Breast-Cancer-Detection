# Breast Cancer Detection via Manually Implemented ML Algorithms

## Overview

This project showcases the implementation of Logistic Regression, Neural Networks, and XGBoost from scratch in Python. External libraries were not used for the core algorithms in order to understand the application of ML concepts via practical application.

The models were tested on the Breast Cancer Wisconsin Diagnostic dataset, and their performance was evaluated in terms of accuracy.

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 92.98%   |
| Neural Network      | 86.84%   |
| XGBoost             | 95.61%   |

## Implemented Models

1. Logistic Regression
   This was the baseline model, implemented using sigmoid activation and gradient descent. A confusion matrix was also created to visualize true positives, true negatives, false positives, and false negatives from the hypothesis function.
   ![Confusion Quadrant for Logistic Regression](/confusion-quadrant.png)

2. Neural Network
   A fully connected neural network with 1 hidden layer comprised of 10 neurons. Details:

- Sigmoid Activation in the Output Layer
- L2 Regularization (Regularization Strength of 0.01)
- Early Stopping (20 Epochs)
- Learning Rate: 0.01
- Max Iterations: 500

3. XGBoost
   A custom implementation using decision trees created from scratch with optimal splits, along with accounting for a regularization parameter. Details:

- Number of Trees: 10
- Learning Rate: 0.1
- Max Depth: 3

## Dataset

The Breast Cancer Wisconsin Diagnostic dataset was used, retrieved from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

## How to Run

### Install Dependencies

Ensure you have Python 3.7+ installed. Install the required libraries using: `pip install -r requirements`

### Run the Project

Execute the 'main.py' script: `python main.py`
