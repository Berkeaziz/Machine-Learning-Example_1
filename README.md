# Machine Learning Algorithms – MATLAB Examples

This repository contains **educational and reference implementations** of fundamental machine learning algorithms in MATLAB.

The implementations are provided as **examples for learning and practice purposes only**.  
They are **not intended to be submitted as homework solutions** and should be used to understand the algorithms, mathematical formulations, and implementation details.

---

## Purpose of This Repository

The main goal of this repository is to:
- Demonstrate how core machine learning algorithms can be implemented **from scratch**
- Provide **worked examples** that help students understand theory–implementation connections
- Serve as a **personal reference archive** for future projects and revisions

All experiments follow the structure and problem formulations commonly used in undergraduate machine learning courses.

---

## Implemented Topics

### 1. Linear Regression (One Variable – L1 Loss)

- Example implementation of linear regression using **Mean Absolute Error (L1 loss)**
- Batch gradient descent with **subgradient method**
- Data visualization and interpretation
- Cost function surface and contour visualization
- Example predictions for unseen inputs

> This section is meant to illustrate how L1 loss behaves differently from L2 loss during optimization.

---

### 2. Linear Regression (Multiple Variables – L2 Loss)

- Example of multivariate linear regression
- Feature normalization and its effect on convergence
- Batch gradient descent implementation
- Learning rate experimentation
- Closed-form solution using **normal equations**

These examples highlight the differences between iterative and analytical solutions.

---

### 3. Logistic Regression (Binary Classification)

- Logistic regression implemented from scratch
- Cost function and gradient computation
- Optimization using MATLAB’s `fminunc`
- Probability estimation for a sample input
- Visualization of decision boundaries

The focus here is on understanding **probabilistic classification** and optimization.

---

### 4. Logistic Regression with Regularization

- Polynomial feature mapping (up to 5th degree)
- Regularized cost function and gradient
- Effect of different regularization strengths (λ)
- Visualization of nonlinear decision boundaries

This section demonstrates how regularization controls model complexity.

---

### 5. Linear Discriminant Analysis (LDA) & Quadratic Discriminant Analysis (QDA)

- Example implementations of LDA and QDA from scratch
- Computation of:
  - Class means
  - Covariance matrices
  - Prior probabilities
- Decision boundary visualization
- Comparison with MATLAB’s built-in `fitcdiscr` function

These examples focus on **generative classification models**.

---

### 6. k-Nearest Neighbors (k-NN)

- k-NN classifier implemented from scratch
- Euclidean distance-based classification
- 5-fold cross-validation example
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Comparison with MATLAB’s `fitcknn`

This section demonstrates instance-based learning and model evaluation.

---

## Notes

- All figures and plots are generated programmatically
- No screenshots are used
- External machine learning libraries are avoided unless explicitly stated
- The code prioritizes **clarity and educational value** over optimization

---




