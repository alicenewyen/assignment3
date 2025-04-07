

---

# Optical Digit Recognition using Perceptron and MLP

This project implements a comparative study between a single-layer Perceptron and a Multi-Layer Perceptron (MLP) for classifying optically recognized handwritten digits. The solution is part of CS 4210 Assignment #3.

## Overview

This Python script uses NumPy and pandas for data manipulation and scikit-learn for building and evaluating two neural network models:

- **Perceptron:** A single-layer classifier.
- **MLPClassifier:** A multi-layer classifier with one hidden layer consisting of 25 neurons using the logistic (sigmoid) activation function.

The goal is to experiment with various hyperparameters (learning rate and whether to shuffle the training data) to find the best performing models on the optdigits dataset.

## Data Loading and Preprocessing

The project uses two datasets:
- `optdigits.tra`: The training set.
- `optdigits.tes`: The test set.

Each instance in the dataset consists of 64 features (representing pixel intensities) and a class label (the digit). The script extracts the first 64 columns as features and the last column as the label.

## Hyperparameter Setup

Two hyperparameters are explored:
- **Learning Rate:** A list of values ranging from 0.0001 to 1.0.
- **Shuffle Option:** A Boolean value indicating whether the training data should be shuffled in each epoch.

The script iterates over all combinations of these hyperparameters.

## Model Training and Evaluation

For each hyperparameter combination:
1. **Model Initialization:**
   - **Perceptron:** Configured with `eta0` (learning rate), `shuffle`, and `max_iter=1000`.
   - **MLPClassifier:** Configured with:
     - Activation function: `logistic`
     - Hidden layer: one hidden layer with 25 neurons (`hidden_layer_sizes=(25,)`)
     - Learning rate: `learning_rate_init` set to the current learning rate
     - Shuffle: current shuffle option
     - Maximum iterations: `max_iter=1000`

2. **Training:**  
   The models are trained using the `fit()` method on the training data.

3. **Prediction:**  
   The script makes predictions for each test sample individually using a sample-by-sample iteration (with `zip()`). This approach helps to illustrate how predictions are made on a per-instance basis.

4. **Accuracy Calculation:**  
   The accuracy is computed by comparing each predicted label with the true label and calculating the percentage of correct predictions.

5. **Best Accuracy Tracking:**  
   The script tracks the highest accuracy obtained for both the Perceptron and the MLP. Whenever a new best accuracy is achieved, it prints out a message with the corresponding hyperparameters.

## How to Run

1. Ensure that the files `optdigits.tra` and `optdigits.tes` are in the same directory as the script.
2. Install the necessary libraries (if not already installed):

   ```bash
   pip install numpy pandas scikit-learn
   ```

3. Run the script using Python:

   ```bash
   python perceptron.py
   ```

## Code Description

Below is a brief explanation of the key sections of the code:

- **Data Loading:**  
  The script reads the training and test data using `pandas.read_csv()`, then extracts the feature matrix (`X_training` and `X_test`) and labels (`y_training` and `y_test`).

- **Hyperparameter Iteration:**  
  The script defines lists of learning rates and shuffle options. It then uses nested loops to iterate through all combinations.

- **Model Creation and Training:**  
  For each hyperparameter combination, the script creates a Perceptron and an MLP model with the specified settings, trains them using the training data, and makes predictions on the test set.

- **Sample-by-Sample Prediction:**  
  Using a `for` loop with `zip()`, the script iterates over each test sample, makes a prediction, and counts the number of correct predictions to calculate the accuracy.

- **Tracking Best Accuracy:**  
  The script updates and prints the best accuracy achieved for each model along with the current hyperparameter settings when a new high is reached.

## Conclusion

This project demonstrates how to build and evaluate two neural network models for digit recognition while experimenting with different hyperparameters. The code is designed to be modular and easily extendable for further experimentation.

---
