# Signature Forgery Detection (CNN)

A simple, end-to-end baseline for signature forgery detection using a CNN in TensorFlow/Keras.

## Dataset structure (expected)

This repo assumes the following folder layout:

```text
sig_data/
  train/
    person_1/
      *.png (or .jpg/.jpeg)
    person_2/
      ...
  validation/
    person_1/
      ...
  test/
    person_1/
      ...
```

**Label rule:** if the filename contains `forg` (case-insensitive), it is treated as **forged (1)**, otherwise **genuine (0)**.

## Quickstart

### 1) Create env + install deps
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Train
```bash
python train.py --data_root /path/to/sig_data --epochs 50 --batch_size 64
```

This saves:
- `artifacts/model.h5`
- `artifacts/metadata.json` (stores resize H/W used)
- `artifacts/accuracy.png`, `artifacts/loss.png`

### 3) Predict on a single image
```bash
python predict.py --model artifacts/model.h5 --metadata artifacts/metadata.json --image /path/to/image.png
```

Output includes a forgery probability and the predicted class.

## Notes
- This is a baseline. Real-world signature verification often benefits from writer-dependent modeling, Siamese/Triplet networks, and careful evaluation protocols.
