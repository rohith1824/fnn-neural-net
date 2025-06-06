import numpy as np
from sklearn.datasets import fetch_openml
from src.layers import Dense
from src.utils import softmax, cross_entropy_loss, softmax_ce_gradient

class NeuralNetwork:
    def __init__(self, layer_sizes):
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
        for epoch in range(1, epochs + 1):
            # Forward
            logits = self.forward(X)                # shape = (N,10)

            # Loss + gradient w.r.t. logits 
            loss, dlogits = self.compute_loss(logits, y)

            # Backward
            self.backward(dlogits)

            self.update_params(lr)

            if verbose and (epoch % max(1, epochs // 10) == 0):
                preds = np.argmax(logits, axis=1)        # shape (N,)
                acc = np.mean(preds == y)
                print(f"Epoch {epoch}/{epochs}  —  loss: {loss:.4f}  —  acc: {acc:.3f}")


    def predict(self, X):
        """
        After training, returns the predicted class (0 to 9) for each row in X.
        """
        logits = self.forward(X)         # shape = (N,10)
        return np.argmax(logits, axis=1) # shape = (N,)


def train_two_layer(
    layer1: Dense,
    layer2: Dense,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    num_epochs=10,
    learning_rate=0.1,
    batch_size=128,
):
    n_train = X_train.shape[0]
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        perm = np.random.permutation(n_train)
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]

        # Mini-batch loop
        for i in range(0, n_train, batch_size):
            X_batch = X_shuf[i : i + batch_size]   # (batch_size, input_dim)
            Y_batch = Y_shuf[i : i + batch_size]   # (batch_size, n_classes)

            # Forward pass
            z1 = layer1.forward(X_batch)         # (batch_size, hidden_dim), ReLU inside
            z2 = layer2.forward(z1)              # (batch_size, n_classes), linear

            probs = softmax(z2)                  # (batch_size, n_classes)

            # Backward pass
            dz2 = softmax_ce_gradient(probs, Y_batch)  # (batch_size, n_classes)
            da1 = layer2.backward(dz2)                     # (batch_size, hidden_dim)
            _   = layer1.backward(da1)                      # (batch_size, input_dim)

            # Update weights
            layer2.W -= learning_rate * layer2.dW
            layer2.b -= learning_rate * layer2.db
            layer1.W -= learning_rate * layer1.dW
            layer1.b -= learning_rate * layer1.db

        # End of epoch, compute full-train & val metrics
        z1_train = layer1.forward(X_train)
        z2_train = layer2.forward(z1_train)
        probs_train = softmax(z2_train)
        train_loss = cross_entropy_loss(probs_train, Y_train)

        z1_val = layer1.forward(X_val)
        z2_val = layer2.forward(z1_val)
        probs_val = softmax(z2_val)
        val_loss = cross_entropy_loss(probs_val, Y_val)

        preds_val = np.argmax(probs_val, axis=1)
        true_val  = np.argmax(Y_val, axis=1)
        val_acc = np.mean(preds_val == true_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch}/{num_epochs}  —  "
            f"train_loss: {train_loss:.4f}  —  "
            f"val_loss: {val_loss:.4f}  —  "
            f"val_acc: {val_acc:.4f}"
        )

    return history


def evaluate_two_layer(layer1: Dense, layer2: Dense, X_test: np.ndarray, Y_test: np.ndarray):
    """
    Runs a forward pass on the test set and returns test accuracy.
    """
    z1 = layer1.forward(X_test)          # (n_test, hidden_dim)
    z2 = layer2.forward(z1)              # (n_test, n_classes)
    probs = softmax(z2)                  # (n_test, n_classes)
    preds = np.argmax(probs, axis=1)
    true_labels = np.argmax(Y_test, axis=1)
    test_acc = np.mean(preds == true_labels)
    return test_acc