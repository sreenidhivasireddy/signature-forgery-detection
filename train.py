import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.metrics import classification_report, confusion_matrix

from utils import compute_mean_hw, load_signature_data, save_metadata

def build_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def plot_history(history, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure()
    plt.plot(acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "accuracy.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss.png"), bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to sig_data folder containing train/validation/test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--artifacts", default="artifacts", help="Output folder for model + metadata")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train")
    val_dir   = os.path.join(args.data_root, "validation")
    test_dir  = os.path.join(args.data_root, "test")

    mean_h, mean_w = compute_mean_hw(train_dir)
    print(f"[info] Resize (H,W) = ({mean_h},{mean_w})")

    X_train, y_train = load_signature_data(train_dir, mean_h, mean_w)
    X_val, y_val     = load_signature_data(val_dir, mean_h, mean_w)
    X_test, y_test   = load_signature_data(test_dir, mean_h, mean_w)

    print("[info] Train:", X_train.shape, "forg%:", float(np.mean(y_train)))
    print("[info] Val:  ", X_val.shape,   "forg%:", float(np.mean(y_val)))
    print("[info] Test: ", X_test.shape,  "forg%:", float(np.mean(y_test)))

    model = build_cnn((mean_h, mean_w, 1))
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print("[result] Test accuracy:", test_acc, "Test loss:", test_loss)

    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs(args.artifacts, exist_ok=True)
    model_path = os.path.join(args.artifacts, "model.h5")
    meta_path  = os.path.join(args.artifacts, "metadata.json")
    model.save(model_path)
    save_metadata(meta_path, mean_h, mean_w)
    plot_history(history, args.artifacts)

    print(f"[saved] {model_path}")
    print(f"[saved] {meta_path}")
    print(f"[saved] {os.path.join(args.artifacts, 'accuracy.png')}")
    print(f"[saved] {os.path.join(args.artifacts, 'loss.png')}")

if __name__ == "__main__":
    main()
