import numpy as np
from src.utils import generate_data, generate_blobs
from src.nn import NeuralNetwork


def main():
    # Generate a simple 2D toy dataset
    X, y = generate_blobs()  

    # Check shapes:
    assert X.ndim == 2 and X.shape[1] == 2, "X must be (N, 2)"
    assert y.ndim == 1 and X.shape[0] == y.shape[0], "y must be (N,)"

    # Instantiate a tiny neural network: 2 → 8 → 2
    model = NeuralNetwork([2, 8, 2])

    # Train for a few epochs, watching the loss decrease.
    epochs = 100
    learning_rate = 0.1

    print(f"\nTraining on toy data: X.shape={X.shape}, y.shape={y.shape}")
    model.train(X, y, epochs=epochs, lr=learning_rate, verbose=True)

    # Run predictions on the same toy data
    y_pred = model.predict(X)  # returns integers 0 or 1

    print("\nFirst 10 examples (toy dataset):")
    for i in range(min(10, X.shape[0])):
        xi = X[i]
        print(f"  X[{i}] = {xi},  true = {y[i]},  pred = {y_pred[i]}")

    # Compute overall training accuracy on toy data

    train_acc = np.mean(y_pred == y)
    print(f"\nToy-data training accuracy: {train_acc:.3f}\n")


if __name__ == "__main__":
    main()
