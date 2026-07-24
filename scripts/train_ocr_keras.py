"""Train the character-recognition CNN used by the ``cnn`` OCR backend.

Expects a directory of class-labelled character images::

    dataset/
        0/ *.jpg
        1/ *.jpg
        ...
        alef/ *.jpg      # letters, one folder per class

Each sub-folder name becomes a label. The trained model is saved to
``models/ocr_cnn.h5`` and the ordered label list to ``models/ocr_cnn.labels.json``
so the backend can map output neurons back to characters.

Usage:
    python scripts/train_ocr.py --data dataset --epochs 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Platrix OCR CNN")
    parser.add_argument("--data", required=True, help="Labelled character dataset dir")
    parser.add_argument("--out", default="models/ocr_cnn.h5")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--height", type=int, default=60)
    args = parser.parse_args()

    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    target = (args.height, args.width)
    gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    common = dict(
        target_size=target,
        batch_size=args.batch,
        class_mode="categorical",
        color_mode="grayscale",
        seed=1,
    )
    train = gen.flow_from_directory(args.data, subset="training", **common)
    valid = gen.flow_from_directory(args.data, subset="validation", **common)

    num_classes = train.num_classes
    model = keras.Sequential(
        [
            layers.Input(shape=(args.height, args.width, 1)),
            layers.Conv2D(32, 4, activation="relu"),
            layers.Conv2D(32, 4, activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    model.fit(train, validation_data=valid, epochs=args.epochs, verbose=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)

    # Persist labels in output-neuron order.
    labels = [None] * num_classes
    for name, idx in train.class_indices.items():
        labels[idx] = name
    out.with_suffix(".labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved model → {out}")
    print(f"Saved labels → {out.with_suffix('.labels.json')}")


if __name__ == "__main__":
    main()
