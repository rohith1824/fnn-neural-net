# imports
import time
import numpy as np
from sklearn.datasets import fetch_openml
from src.utils import train_val_test_split
from src.nn import NeuralNetwork

SEED           = 0
TRAIN_FRACTION = 0.80   # 80 % train, 10 % val, 10 % test
EPOCHS         = 50
LR             = 0.1
ARCH           = [784, 128, 64, 10] 

# Load data                                                         
def load_data(name="mnist", flatten=True):
    """
    name: "mnist" or "fashion_mnist"
    returns X in [0,1] shape=(n_samples,784), y as int labels
    """
    name = name.lower()
    if name == "mnist":
        ds_id = "mnist_784"
        msg   = "MNIST"
    elif name in ("fashion", "fashion_mnist"):
        ds_id = "Fashion-MNIST"
        msg   = "Fashion-MNIST"
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    print(f"📥  Fetching {msg}…")
    ds = fetch_openml(ds_id, version=1, as_frame=False, cache=True)
    X  = ds["data"].astype(np.float32) / 255.0
    y  = ds["target"].astype(int)

    if not flatten:
        X = X.reshape(-1, 28, 28)
    return X, y

# Train model
def main(dataset="mnist"):
    np.random.seed(SEED)

    # dataset
    X, y = load_data(dataset)
    X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(
        X, y,
        train_frac=TRAIN_FRACTION,
        val_frac=0.10,
        test_frac=0.10,
        random_seed=SEED,
    )

    # model
    model = NeuralNetwork(ARCH)

    # train
    t0 = time.time()
    model.train(
        X_tr, y_tr,
        epochs=EPOCHS,        # an upper bound; training may stop sooner
        lr=LR,
        batch_size=128,
        X_val=X_val,
        y_val=y_val,
        patience=3,           # stop after 6 epochs with < min_delta improvement
        min_delta=0.01,
        verbose=True,
    )
    train_secs = time.time() - t0


    # validate / test
    val_acc  = np.mean(model.predict(X_val) == y_val)
    test_acc = np.mean(model.predict(X_te) == y_te)

    # summary
    print("\n===== MNIST BENCH SUMMARY =====")
    print(f"Architecture : {ARCH}")
    print(f"Epochs       : {EPOCHS}")
    print(f"LR           : {LR}")
    print(f"Train time   : {train_secs:.1f} s")
    print(f"Val accuracy : {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
