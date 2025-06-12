import numpy as np
from src.layers import Dense
from src.utils import softmax, cross_entropy_loss, softmax_ce_gradient, one_hot_encode

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
        """
        Compute softmax cross-entropy loss and its gradient.
        """
        # Softmax numerically stable
        # Shift by row-wise max to avoid overflow
        shifted = logits - np.max(logits, axis=1, keepdims=True)  # (N,10)
        exp_scores = np.exp(shifted)                               # (N,10)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N,10)

        # Cross-entropy
        N = logits.shape[0]
        # pick the probability assigned to the true class for each sample
        correct_logprobs = -np.log(probs[np.arange(N), y_true])
        loss = np.mean(correct_logprobs)

        # Gradient of loss w.r.t. logits:
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


    def train(
            self,
            X, y,
            epochs: int = 30,
            lr: float = 0.1,
            batch_size: int = 128,
            shuffle: bool = True,
            verbose: bool = True,
            *,
            X_val=None,
            y_val=None,
            patience: int = 5,
            min_delta: float = 1e-3,
        ):
            """
            Train the network with mini-batch SGD, optional validation, and early stopping
            """
            n_samples = X.shape[0]
            best_val_loss = np.inf
            epochs_bad    = 0
            # store best weights so we can roll back
            best_weights = [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

            for epoch in range(1, epochs + 1):
                if shuffle:
                    perm = np.random.permutation(n_samples)
                    X, y = X[perm], y[perm]

                # mini-batch SGD
                for i in range(0, n_samples, batch_size):
                    xb, yb = X[i : i + batch_size], y[i : i + batch_size]
                    logits     = self.forward(xb)
                    loss, dlog = self.compute_loss(logits, yb)
                    self.backward(dlog)
                    self.update_params(lr)

                # Model metrics
                logits_tr = self.forward(X)
                train_acc = (np.argmax(logits_tr, axis=1) == y).mean()

                if X_val is not None:
                    logits_val = self.forward(X_val)
                    probs_val  = softmax(logits_val)
                    val_loss   = cross_entropy_loss(probs_val, one_hot_encode(y_val, 10))
                    val_acc    = (np.argmax(logits_val, axis=1) == y_val).mean()

                    if verbose:
                        print(f"Epoch {epoch}/{epochs} "
                            f"— train_acc {train_acc:.3f} "
                            f"— val_loss {val_loss:.4f} "
                            f"— val_acc {val_acc:.3f}")

                    # Early callback check
                    if best_val_loss - val_loss > min_delta:
                        best_val_loss = val_loss
                        epochs_bad    = 0
                        best_weights  = [(l.W.copy(), l.b.copy()) for l in self.layers]
                    else:
                        epochs_bad += 1
                        if epochs_bad >= patience:
                            print(f"\n⏹️  Early stopping at epoch {epoch} "
                                f"(no val loss drop >{min_delta} for {patience} epochs)")
                            # restore best
                            for (W_best, b_best), layer in zip(best_weights, self.layers):
                                layer.W[:] = W_best
                                layer.b[:] = b_best
                            break
                else:
                    # no validation set provided
                    if verbose:
                        print(f"Epoch {epoch}/{epochs} — train_acc {train_acc:.3f}")


    def predict(self, X):
        """
        After training, returns the predicted class (0 to 9) for each row in X.
        """
        logits = self.forward(X)         # shape = (N,10)
        return np.argmax(logits, axis=1) # shape = (N,)
