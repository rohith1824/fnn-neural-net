# Neural Network From Scratch

## 📌 Description
This project implements a **feedforward neural network from scratch** using only NumPy—**no PyTorch or TensorFlow**. The goal is to build every component manually (weight initialization, forward pass, backward pass, loss computation, and parameter updates) so you deeply understand the math and mechanics behind training a neural network. Along the way, you’ll write clean, modular code, include unit tests for core functions, and visualize training dynamics.

---

## 🎯 Goals

- [ ] **Weight & bias initialization**  
  - Implement functions to initialize parameters for each layer (e.g., small random values, Xavier/He initialization).
- [ ] **Activation functions**  
  - Code Sigmoid, ReLU, and (optionally) Tanh, along with their derivatives.
- [ ] **Forward pass (single layer)**  
  - Compute `Z = W·X + b` and apply activation `A = ϕ(Z)`.
- [ ] **Forward pass (multi-layer)**  
  - Stack multiple linear+activation layers to compute final output `ŷ`.
- [ ] **Loss functions**  
  - Implement Mean Squared Error (MSE) and/or Cross-Entropy loss, with forward and backward computations.
- [ ] **Backward pass (single layer)**  
  - Derive and code gradients `dW`, `db`, and `dX` for a single linear+activation layer.
- [ ] **Backward pass (full network)**  
  - Chain derivatives through all layers to compute gradients w.r.t. every parameter.
- [ ] **Gradient descent optimizer**  
  - Implement parameter updates with a fixed learning rate (and, optionally, add momentum or learning rate decay).
- [ ] **Training loop**  
  - Set up mini-batch or full-batch looping over epochs, compute forward → loss → backward → update for N epochs.
- [ ] **Performance monitoring & visualization**  
  - Plot training/validation loss over epochs and track accuracy on a simple dataset (e.g., toy dataset or a small CSV).
- [ ] **Evaluate on a real dataset**  
  - Load a simple dataset (e.g., MNIST subset, Iris, or a synthetic problem), train your network, and report final accuracy.
- [ ] **Unit tests**  
  - Write tests for initialization, forward pass, backward pass, and loss functions to ensure gradients are correct (e.g., via numerical gradient check).
- [ ] **Code modularity & documentation**  
  - Organize code into modules (e.g., `layers/`, `activations/`, `losses/`, `utils/`), add docstrings, and keep functions single-purpose.

---