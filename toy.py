import numpy as np
from src.utils import generate_data, generate_blobs
from src.nn import NeuralNetwork


def main():
    # -----------------------------------------------------------------------
    # 1) Generate a simple 2D toy dataset using utils.load_toy_data()
    #
    #    This function should return:
    #      X: shape (N, 2), numpy array of 2D points
    #      y: shape (N,),  integer labels in {0, 1}
    #
    #    For example, load_toy_data() might produce a linearly separable
    #    2-class dataset, or identity mapping simplified to two clusters.
    # -----------------------------------------------------------------------
    X, y = generate_blobs()  

    # Check shapes:
    assert X.ndim == 2 and X.shape[1] == 2, "X must be (N, 2)"
    assert y.ndim == 1 and X.shape[0] == y.shape[0], "y must be (N,)"

    # -----------------------------------------------------------------------
    # 2) Instantiate a tiny neural network: 2 → 8 → 2
    #
    #    Here, output_dim = 2, so we treat it as a 2-class classification.
    #    Internally, the final layer will produce 2 logits, and
    #    softmax+cross-entropy will be applied during training.
    # -----------------------------------------------------------------------
    model = NeuralNetwork([2, 8, 2])

    # -----------------------------------------------------------------------
    # 3) Train for a few epochs, watching the loss decrease.
    #
    #    We choose a modest number of epochs (e.g., 100) and a learning rate.
    #    The train() method will print loss (and accuracy) at intervals.
    # -----------------------------------------------------------------------
    epochs = 100
    learning_rate = 0.1

    print(f"\nTraining on toy data: X.shape={X.shape}, y.shape={y.shape}")
    model.train(X, y, epochs=epochs, lr=learning_rate, verbose=True)

    # -----------------------------------------------------------------------
    # 4) After training, run predictions on the same toy data
    #    and print out the first few examples: (X_i, true_label, pred_label).
    # -----------------------------------------------------------------------
    y_pred = model.predict(X)  # returns integers 0 or 1

    print("\nFirst 10 examples (toy dataset):")
    for i in range(min(10, X.shape[0])):
        xi = X[i]
        print(f"  X[{i}] = {xi},  true = {y[i]},  pred = {y_pred[i]}")

    # -----------------------------------------------------------------------
    # 5) (Optional) Compute overall training accuracy on toy data
    # -----------------------------------------------------------------------
    train_acc = np.mean(y_pred == y)
    print(f"\nToy-data training accuracy: {train_acc:.3f}\n")


if __name__ == "__main__":
    main()
