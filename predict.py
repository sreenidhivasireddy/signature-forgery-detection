import argparse
import cv2
import numpy as np
import tensorflow as tf

from utils import load_metadata

def preprocess_image(path: str, h: int, w: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (w, h))
    x = (img.astype(np.float32) / 255.0)[None, ..., None]  # (1,H,W,1)
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.h5")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--image", required=True, help="Path to signature image")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    h, w = load_metadata(args.metadata)
    model = tf.keras.models.load_model(args.model)

    x = preprocess_image(args.image, h, w)
    prob_forg = float(model.predict(x, verbose=0)[0][0])
    pred = 1 if prob_forg >= args.threshold else 0

    print(f"Forgery probability: {prob_forg:.4f}")
    print("Prediction:", "FORGED (1)" if pred == 1 else "GENUINE (0)")

if __name__ == "__main__":
    main()
