import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.nn import NeuralNetwork
from src.utils import generate_blobs

def test_classification_loss_decreases():
    """
    Build a tiny 2‐layer NN on a synthetic binary‐blobs dataset.
    Check that cross‐entropy loss (via compute_loss) decreases after a few epochs.
    """
    np.random.seed(0)
    X, y = generate_blobs(n_samples=100, random_seed=0)

    # 2→4(ReLU)→2(logits)
    model = NeuralNetwork([2, 4, 2])

    logits_initial = model.forward(X)
    loss_initial, _ = model.compute_loss(logits_initial, y)

    model.train(X, y, epochs=5, lr=0.1, verbose=False)

    logits_final = model.forward(X)
    loss_final, _ = model.compute_loss(logits_final, y)

    assert loss_final < loss_initial, (
        f"Expected loss to decrease: {loss_initial:.4f} → {loss_final:.4f}"
    )
