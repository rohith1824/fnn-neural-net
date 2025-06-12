import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.layers import Dense

def test_dense_forward_shape():
    np.random.seed(0)
    batch_size = 4
    n_input = 5
    n_output = 3
    x = np.random.randn(batch_size, n_input)

    dense = Dense(n_input, n_output, activation='linear')
    out = dense.forward(x)

    assert out.shape == (batch_size, n_output), (
        f"Expected output shape {(batch_size, n_output)}, got {out.shape}"
    )

def numerical_gradient_dense(dense, x, dout, eps=1e-5):
    """
    Numerically approximate dW and db for a single Dense layer
    by finite differences.
    """
    W_orig = dense.W.copy()
    b_orig = dense.b.copy()
    dense.forward(x)

    dW_num = np.zeros_like(dense.W)
    db_num = np.zeros_like(dense.b)

    # Numerical dW
    for i in range(dense.W.shape[0]):
        for j in range(dense.W.shape[1]):
            dense.W[i, j] = W_orig[i, j] + eps
            plus_out = dense.forward(x).copy()
            dense.W[i, j] = W_orig[i, j] - eps
            minus_out = dense.forward(x).copy()
            dense.W[i, j] = W_orig[i, j]
            dW_num[i, j] = np.sum((plus_out - minus_out) * dout) / (2 * eps)

    # Numerical db
    for j in range(dense.b.shape[1]):
        dense.b[0, j] = b_orig[0, j] + eps
        plus_out = dense.forward(x).copy()
        dense.b[0, j] = b_orig[0, j] - eps
        minus_out = dense.forward(x).copy()
        dense.b[0, j] = b_orig[0, j]
        db_num[0, j] = np.sum((plus_out - minus_out) * dout) / (2 * eps)

    return dW_num, db_num

def test_dense_backward_gradients():
    np.random.seed(0)
    x = np.random.randn(3, 4)
    dout = np.random.randn(3, 2)
    dense = Dense(input_dim=4, output_dim=2, activation='linear')

    _ = dense.forward(x)
    _ = dense.backward(dout)
    dW_analytic = dense.dW.copy()
    db_analytic = dense.db.copy()

    dW_num, db_num = numerical_gradient_dense(dense, x, dout, eps=1e-5)

    assert np.allclose(dW_analytic, dW_num, atol=1e-5), (
        f"dW mismatch:\nanalytic:\n{dW_analytic}\nnumeric:\n{dW_num}"
    )
    assert np.allclose(db_analytic, db_num, atol=1e-5), (
        f"db mismatch:\nanalytic:\n{db_analytic}\nnumeric:\n{db_num}"
    )
