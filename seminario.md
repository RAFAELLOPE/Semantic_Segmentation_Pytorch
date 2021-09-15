---
marp: true
---

<H1> Semantic Segmentation Pytorch </H1>

## Schema
1. CRISP-DM
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

## 4. Modeling - Stochastic Gradient Descent

Optimization problem in Machine Learning boils down to:

$D = \{ (x_{i}, y_{i}) \}_{i = 1}^{n} \in \R^{d} \times Y$

where *d* = dimension and *n* = observations.

Least-Squares:

$\frac{1}{n}\| Ax - b \|_{2}^{2} = \frac{1}{n}\sum_{i = 1}^{n}(a_{i}^{T}x - b_{i})^{2}$

$\frac{1}{n}\sum_{i = 1}^{n}(a_{i}^{T}x - b_{i})^{2} + \lambda\sum_{j = 1}^{d}|x_{j}|$

$\frac{1}{n}\sum_{i = 1}^{n}(a_{i}^{T}x - b_{i})^{2} + \lambda\sum_{j = 1}^{d}|x_{j}|^{2}$



Support Vector Machine

$\frac{1}{2} \| x \|^{2} + \frac{c}{n}\sum_{i = 1}^{n} max[ 0, 1 - y_{i}(x^{T}a_{i} + b)]$

Deep Neural Networks
$\frac{1}{n}\sum_{i = 1}^{n}L(y_{i}, DNN(x, a_{i}))$



