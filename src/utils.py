import numpy as np

def generate_data(n_samples=100, n_features=2, n_classes=2, random_seed=None):
    """
    Generate a small random dataset from standard normal distribution.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # random normal
    X = np.random.randn(n_samples, n_features)

    # Labels: integers 0..(n_classes-1), assigned uniformly at random
    y = np.random.randint(low=0, high=n_classes, size=n_samples)

    return X, y

def generate_blobs(n_samples=100, random_seed=None):
    """
    Two Gaussian clusters:
      - Class 0 centered at (1,1)
      - Class 1 centered at (-1,-1)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    half = n_samples // 2
    X0 = np.random.randn(half, 2)*0.2 + np.array([1.0, 1.0])
    y0 = np.zeros(half, dtype=int)

    X1 = np.random.randn(half, 2)*0.2 + np.array([-1.0, -1.0])
    y1 = np.ones(half, dtype=int)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    return X, y



def normalize_data(X, mean=None, std=None):
    """
    Normalize (standardize) inputs to zero mean and unit variance.
    If mean and std are provided, use them; otherwise compute from X.
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        # Avoid division by zero for constant features
        std[std == 0] = 1.0

    X_norm = (X - mean) / std
    return X_norm, mean, std


def one_hot_encode(y, num_classes=None):
    """
    One-hot encode integer labels.
    """
    y = np.asarray(y, dtype=int).ravel()
    if num_classes is None:
        num_classes = int(y.max()) + 1

    n_samples = y.shape[0]
    Y_oh = np.zeros((n_samples, num_classes), dtype=np.float32)
    Y_oh[np.arange(n_samples), y] = 1.0
    return Y_oh


def train_val_test_split(X, y, train_frac=0.7, val_frac=0.15, test_frac=0.15, shuffle=True, random_seed=None):
    """
    Split data into train / validation / test sets.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)

    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    # The remainder goes to test
    n_test = n_samples - n_train - n_val

    idx_train = indices[:n_train]
    idx_val = indices[n_train : n_train + n_val]
    idx_test = indices[n_train + n_val :]

    X_train = X[idx_train]
    y_train = y[idx_train]
    X_val = X[idx_val]
    y_val = y[idx_val]
    X_test = X[idx_test]
    y_test = y[idx_test]

    return X_train, y_train, X_val, y_val, X_test, y_test


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Row-wise softmax.
    z : (batch_size, n_classes)
    returns : (batch_size, n_classes)
    """
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Averaged cross-entropy loss for one-hot labels.
    probs  : (batch_size, n_classes)  –– softmax probabilities
    y_true : (batch_size, n_classes)  –– one-hot ground truth
    """
    m = y_true.shape[0]
    log_lik = np.log(probs + 1e-8)        # avoid log(0)
    return -np.sum(y_true * log_lik) / m


def softmax_ce_gradient(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Gradient of CE loss w.r.t. the logits before softmax.
    probs  : (batch_size, n_classes)
    y_true : (batch_size, n_classes)
    returns: (batch_size, n_classes)
    """
    m = y_true.shape[0]
    return (probs - y_true) / m