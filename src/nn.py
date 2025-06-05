import numpy as np
from src.layers import Dense


class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of ints, e.g. [784, 128, 64, 10]
          - 784   = input_dim (28×28 flattened)
          - 128   = first hidden layer size
          - 64    = second hidden layer size
          - 10    = number of classes (MNIST digits 0–9)

        We will build:
          Dense(784→128, relu)
          Dense(128→64, relu)
          Dense(64→10, linear)  # raw logits; softmax done externally
        """
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]

            if i < len(layer_sizes) - 2:
                # hidden layer → use ReLU
                self.layers.append(Dense(in_dim, out_dim, activation='relu'))
            else:
                # final layer → produce raw logits
                self.layers.append(Dense(in_dim, out_dim, activation='linear'))


    def forward(self, x):
        """
        x: np.ndarray, shape = (batch_size, 784)
        Returns: logits, shape = (batch_size, 10)  (no softmax yet)
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out


    def compute_loss(self, logits, y_true):
        """
        1) Apply softmax to logits → probabilities (batch,10)
        2) Compute cross-entropy loss
        3) Return (loss, dlogits)

        Arguments:
          - logits: np.ndarray, shape = (N, 10)
          - y_true: np.ndarray of integers, shape = (N,)
                    Each entry ∈ {0,1,…,9}

        Returns:
          - loss: scalar average cross-entropy over batch
          - dlogits: ∂L/∂logits, shape = (N,10)
        """
        # 1) Softmax (numerically stable)
        #    Shift by row-wise max to avoid overflow
        shifted = logits - np.max(logits, axis=1, keepdims=True)  # (N,10)
        exp_scores = np.exp(shifted)                               # (N,10)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N,10)

        # 2) Cross-entropy
        N = logits.shape[0]
        # pick the probability assigned to the true class for each sample
        correct_logprobs = -np.log(probs[np.arange(N), y_true])
        loss = np.mean(correct_logprobs)

        # 3) Gradient of loss w.r.t. logits:
        # For softmax + CE, dL/d( z_i ) = (p_i - 1_{i=y_true}) / N
        dlogits = probs.copy()                   # (N,10)
        dlogits[np.arange(N), y_true] -= 1        # subtract 1 at true‐class positions
        dlogits /= N

        return loss, dlogits


    def backward(self, dlogits):
        """
        dlogits: ∂L/∂logits from compute_loss, shape = (N,10)
        We simply pass that gradient backward through each layer
        in reverse order. Each Dense.backward(dout) sets its .dW/.db
        and returns dx for the previous layer.
        """
        grad = dlogits
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        # no return needed; parameter gradients are stored inside each Dense


    def update_params(self, lr):
        """
        Simple SGD step: W ← W − lr·dW,  b ← b − lr·db  for every Dense layer.
        """
        for layer in self.layers:
            # Only Dense has weights/biases:
            if isinstance(layer, Dense):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db


    def train(self, X, y, epochs=10, lr=0.01, verbose=True):
        """
        Training loop over the entire training set (full‐batch GD).
        Arguments:
          - X: np.ndarray, shape = (num_samples, 784)
          - y: np.ndarray of ints, shape = (num_samples,) with values in [0..9]
          - epochs: how many full passes over X
          - lr: learning rate for SGD
        """
        for epoch in range(1, epochs + 1):
            # --- FORWARD ---
            logits = self.forward(X)                # shape = (N,10)

            # --- LOSS + GRADIENT w.r.t. logits ---
            loss, dlogits = self.compute_loss(logits, y)

            # --- BACKWARD ---
            self.backward(dlogits)

            # --- UPDATE parameters ---
            self.update_params(lr)

            if verbose and (epoch % max(1, epochs // 10) == 0):
                # Optionally compute train‐accuracy for sanity check
                preds = np.argmax(logits, axis=1)        # shape (N,)
                acc = np.mean(preds == y)
                print(f"Epoch {epoch}/{epochs}  —  loss: {loss:.4f}  —  acc: {acc:.3f}")


    def predict(self, X):
        """
        After training, returns the predicted class (0 to 9) for each row in X.
        """
        logits = self.forward(X)         # shape = (N,10)
        return np.argmax(logits, axis=1) # shape = (N,)
