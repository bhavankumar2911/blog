---
title: "The Kernel Trick"
date: 2026-08-26
draft: false
tags: ["machine learning", "SVM", "mathematics"]
categories: ["ML Theory"]
summary: "How the kernel trick lets SVMs find non-linear boundaries without ever computing high-dimensional feature vectors."
---

## Introduction

The kernel trick is one of the most elegant ideas in machine learning. It lets us train Support Vector Machines (and other algorithms) on data that is not linearly separable — without ever explicitly computing the coordinates of points in a high-dimensional feature space.

## The Problem with Non-Linear Data

Suppose we have two classes of data that cannot be separated by a straight line (or a hyperplane in higher dimensions). The natural fix is to map the data into a higher-dimensional space where a linear separator does exist.

Given a point $\mathbf{x} \in \mathbb{R}^d$, we apply a feature map:

$$\phi: \mathbb{R}^d \rightarrow \mathbb{R}^D, \quad D \gg d$$

Then we find a linear separator in $\mathbb{R}^D$. The problem: computing $\phi(\mathbf{x})$ explicitly can be astronomically expensive — or even infinite-dimensional.

## The Key Observation

Look at the SVM dual formulation. The decision function is:

$$f(\mathbf{x}) = \sum_{i=1}^{n} \alpha_i y_i \, \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}) + b$$

Notice that $\phi$ only ever appears as a **dot product** $\phi(\mathbf{x}_i)^\top \phi(\mathbf{x})$.

This is the insight: if we can compute that dot product cheaply — without computing $\phi$ itself — we are done.

## The Kernel Function

A **kernel** is a function $k$ such that:

$$k(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^\top \phi(\mathbf{z})$$

We evaluate the dot product in the high-dimensional space using only the original inputs $\mathbf{x}$ and $\mathbf{z}$.

### Common Kernels

| Kernel | Formula |
|--------|---------|
| Linear | $k(\mathbf{x}, \mathbf{z}) = \mathbf{x}^\top \mathbf{z}$ |
| Polynomial | $k(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^\top \mathbf{z} + c)^p$ |
| RBF / Gaussian | $k(\mathbf{x}, \mathbf{z}) = \exp\!\left(-\dfrac{\|\mathbf{x} - \mathbf{z}\|^2}{2\sigma^2}\right)$ |

## A Concrete Example

Consider the polynomial kernel $k(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^\top \mathbf{z})^2$ for $\mathbf{x}, \mathbf{z} \in \mathbb{R}^2$.

Expanding:

$$k(\mathbf{x}, \mathbf{z}) = (x_1 z_1 + x_2 z_2)^2 = x_1^2 z_1^2 + 2x_1 x_2 z_1 z_2 + x_2^2 z_2^2$$

This equals $\phi(\mathbf{x})^\top \phi(\mathbf{z})$ where:

$$\phi(\mathbf{x}) = \begin{bmatrix} x_1^2 \\ \sqrt{2}\, x_1 x_2 \\ x_2^2 \end{bmatrix}$$

We mapped $\mathbb{R}^2 \to \mathbb{R}^3$ implicitly, and the kernel computed the dot product in $\mathbb{R}^3$ with only $O(d)$ operations instead of $O(D)$.

## The RBF Kernel and Infinite Dimensions

The RBF kernel is even more striking. Its corresponding feature map $\phi$ is **infinite-dimensional** — yet the kernel evaluates the dot product in that space as a single exponential. This is computationally free, yet geometrically powerful.

## When Is a Function a Valid Kernel?

By Mercer's theorem, a symmetric function $k(\mathbf{x}, \mathbf{z})$ is a valid kernel if and only if its **Gram matrix** is positive semi-definite for any set of points:

$$K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) \succeq 0$$

## Implementation

```python
import numpy as np

def rbf_kernel(X, Z, sigma=1.0):
    # Compute pairwise squared distances
    sq_dists = (
        np.sum(X**2, axis=1, keepdims=True)
        + np.sum(Z**2, axis=1)
        - 2 * X @ Z.T
    )
    return np.exp(-sq_dists / (2 * sigma**2))

# Example
X = np.array([[1, 2], [3, 4], [5, 6]])
K = rbf_kernel(X, X)
print(K)
```

## Summary

The kernel trick works because:

1. Many ML algorithms (SVM, PCA, regression) only need dot products between data points, never the coordinates themselves.
2. A kernel function computes that dot product implicitly in a high- (or infinite-) dimensional space.
3. This gives us non-linear power at linear computational cost.

The trick is not just a computational shortcut — it fundamentally changes what problems are tractable.
