# Artificial Neuron From Scratch

A simple implementation of a single artificial neuron using Python, NumPy, and gradient descent.

The goal of this project is to understand how binary classification works internally without using a machine-learning model such as `LogisticRegression` from scikit-learn.

## Project Overview

The model takes two input features and predicts one of two classes.

The implementation includes:

* Sigmoid activation function
* Binary cross-entropy loss
* Gradient calculation
* Gradient descent
* Binary classification
* Decision boundary visualization
* 3D visualization of the predictions
* Visualization of the sigmoid function
* Visualization of the loss during training
* Animation of the learning process

## Dataset

The dataset is generated using `make_blobs` from scikit-learn.

```python
X, y = make_blobs(
    n_samples=100,
    n_features=2,
    centers=2,
    random_state=0
)
```

The dataset contains 100 samples with two features and two classes.

## Mathematical Model

The neuron first calculates:

$$
Z = XW + b
$$

Then the sigmoid function converts the result into a probability:

$$
A = \frac{1}{1 + e^{-Z}}
$$

A prediction is made using a threshold of 0.5:

$$
\hat{y} =
\begin{cases}
1 & A \geq 0.5 \\
0 & A < 0.5
\end{cases}
$$

## Cost Function

The model uses binary cross-entropy:

$$
J = -\frac{1}{m}\sum
\left[
y\log(A) + (1-y)\log(1-A)
\right]
$$

The objective of gradient descent is to minimize this cost.

## Gradient Descent

The parameters are updated using:

$$
W = W - \alpha dW
$$

$$
b = b - \alpha db
$$

where `α` is the learning rate.

## Visualizations

The project visualizes:

* The original dataset
* The decision boundary
* The sigmoid function
* The evolution of the cost function
* The 3D probability surface
* The training process through animation

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/artificial-neuron-from-scratch.git
cd artificial-neuron-from-scratch
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Python script:

```bash
python artificial_neuron.py
```

## Technologies

* Python
* NumPy
* Matplotlib
* Scikit-learn
* Plotly

## Learning Objective

This project was created to understand the fundamentals of a neural network and logistic regression by implementing the main components manually.

Rather than using a ready-made classification model, the project implements the forward propagation, loss calculation, gradients, and parameter updates from scratch.
