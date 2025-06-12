# Neural Network From Scratch

## 📌 Description
A lightweight, from-scratch implementation of a feedforward neural network using only NumPy—no PyTorch, no TensorFlow. Every component (parameter initialization, forward pass, backward pass, loss, and optimization) is handwritten to demonstrate a deep understanding of the math and mechanics behind training neural networks. Clean, modular code is paired with unit tests and visualizations.

---

## 🚀 Features

- **Modular Layers**  
  - `Dense` layer supporting ReLU, Sigmoid, and Linear activations  
  - Built-in caching for inputs, pre-activations (Z), and outputs

- **Flexible Architectures**  
  - Construct any `[input,…,output]` topology via `NeuralNetwork([…])`

- **Loss & Metrics**  
  - Numerically-stable Softmax + Cross-Entropy  
  - Training/validation accuracy tracking

- **Optimizers & Callbacks**  
  - Mini-batch SGD with fixed learning rate  
  - Early stopping with patience and weight rollback

- **Data Utilities**  
  - Gaussian blobs and (Fashion-)MNIST loaders  
  - Standardization, one-hot encoding, and train/val/test splits

- **Visualization & Benchmarking**  
  - Plot training/validation loss curves  
  - MNIST benchmark script with timing and accuracy summary

- **Test-Driven Development**  
  - Shape checks, numerical gradient tests, end-to-end loss-decrease tests

---

## 🧮 Math & Mechanics

1. **Parameter Initialization**  
   - **Xavier** (linear/sigmoid):  
     $$W \sim \mathcal{N}\bigl(0,\,\tfrac1{\text{fan\_in}}\bigr)$$
   - **He** (ReLU):  
     $$W \sim \mathcal{N}\bigl(0,\,\tfrac2{\text{fan\_in}}\bigr)$$

2. **Forward Pass**  
   - **Linear**: $Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$
   - **Activations**:  
     - ReLU: $A = \max(0, Z)$
     - Sigmoid: $\sigma(Z)=\tfrac1{1+e^{-Z}}$

3. **Softmax & Cross-Entropy**  
   - Softmax: $p_i = \tfrac{e^{z_i - \max_j z_j}}{\sum_k e^{z_k - \max_j z_j}}$
   - Loss: $L = -\tfrac1N\sum_{n=1}^N \log p_{n,y_n}$

4. **Backward Pass**  
   - Activation gradients (e.g. $\tfrac{d\text{ReLU}}{dZ}=1_{Z>0}$)
   - Parameter gradients:  
     $$dW^{[l]} = (A^{[l-1]})^T\,dZ^{[l]},\quad db^{[l]} = \sum_n dZ_n^{[l]}$$
   - Propagate: $dA^{[l-1]} = dZ^{[l]} (W^{[l]})^T$

5. **Optimization (SGD)**  
   $$W \gets W - \eta\,dW,\quad b \gets b - \eta\,db$$

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/neural-net-from-scratch.git
cd neural-net-from-scratch
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

## 🎯 Quick Start

1. **Toy data**

```bash
python benchmark.py
```

2. **MNIST training**

```bash
python main.py --dataset mnist
```

3. **Plot results** After training, check console summary or hook into `history` for custom plots.

## 📂 Code Structure

```
├── src/
│   ├── layers.py        # Dense layer implementation
│   ├── nn.py            # NeuralNetwork class
│   └── utils.py         # data loaders, preprocessing, loss, metrics
├── tests/
│   ├── test_layers.py   # shape & gradient checks
│   └── test_nn.py       # end-to-end loss-decrease test
├── benchmark.py         # two-layer training helper
├── main.py              # MNIST/Fashion-MNIST pipeline
└── requirements.txt
```

## 📈 Results

* **Gaussian blobs**: 100% train accuracy in <100 epochs
* **MNIST** (50 epochs, early stopping):
   * ⏱ ~15s on CPU
   * 🎯 Val/Test accuracy ≈ 94-97%

## ✅ Testing

```bash
pytest -q
```

All tests should pass, verifying shapes, gradients, and training dynamics.

## 📜 License

MIT © Rohith Senthil Kumar