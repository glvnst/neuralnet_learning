#!/usr/bin/env python3
""" neural_addition.py - an experiment to learn about neural nets

here we're exploring a single-layer, single-neuron neural "net" which is
trained to sum 2 addends. the ideal input weights are 1.0, 1.0 with 0 bias.
it's interesting to explore how easy it is to get this simple job wrong,
especially when you're not at all knowledgeable in the field
"""
import argparse
import csv
import gc
import os
import random
from typing import Any, Optional

import numpy as np
import tensorflow as tf

domain = -65_534, 65_534
randint = random.SystemRandom().randint


def train_model(save_path: Optional[str] = None) -> Any:
    """
    Train the neural network model and return it
    """

    # Generate a set of training data
    x_train = np.random.uniform(*domain, (7_000, 2))
    y_train = np.sum(x_train, axis=1)

    # Generate a set of validation data
    x_val = np.random.uniform(*domain, (3_000, 2))
    y_val = np.sum(x_val, axis=1)

    # Define the neural network model
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(1, input_dim=2, activation="linear")]
    )

    # Compile the model
    model.compile(
        loss="mse",
        optimizer=tf.optimizers.Adam(learning_rate=0.00001),
    )

    # Define the early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=100_000,
        verbose=1,
        callbacks=[early_stopping],
    )

    if save_path:
        model.save(save_path)

    return model


def modelmania() -> None:
    """
    keep training new models with the same settings (except for the rand seed),
    save them in the modelmania subdir, see how their output differs from the
    actual truth, record everything in a csv file, only keep models files that
    are improvements
    """

    output_dir = "./modelmania/"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "modelmania.csv")
    fieldnames = ["sequence", "modelfile", "diff"]
    best_diff = 65535.0

    with open(log_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect="unix")
        writer.writeheader()

        for i in range(50_000):
            model = train_model()
            diff = test_model(model)
            modelfile = f"modelmania/modelmania_{i:05}.keras"
            results = {
                "sequence": f"{i}",
                "modelfile": modelfile,
                "diff": diff,
            }
            writer.writerow(results)
            print(results)

            if diff < best_diff:
                print(f"saving new best model {modelfile} with diff {diff}")
                model.save(modelfile)
                best_diff = diff

            # Explicitly delete the model
            del model

            # Clear the TensorFlow session
            tf.keras.backend.clear_session()

            # python gc
            gc.collect()


def test_model(model: Any) -> float:
    """
    use the model to perform additions, compare the results to the actual true
    values, return the difference between the two
    """
    x_test = np.array([[randint(*domain), randint(*domain)] for _ in range(100_000)])
    y_test = model.predict(x_test)
    # here I get to really showcase my lack of numpy/pandas skill
    diff = sum([abs(x[0] + x[1] - y_test[i][0]) for [i, x] in enumerate(x_test)])
    return diff


def load_model(model_file: str) -> Any:
    """
    load the keras model at the give path on disk and return it
    """
    model = tf.keras.models.load_model(model_file)
    model.summary()
    return model


def main() -> None:
    """
    Train and use the model to predict outputs for test inputs.
    """
    argp = argparse.ArgumentParser(
        description="Train and use a model for simple addition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argp.add_argument(
        "--model-file",
        default="addition_model.keras",
        type=str,
        help="Name of the model file",
    )
    argp.add_argument(
        "--mania",
        action="store_true",
        help="build thousands of models and test them",
    )
    argp.add_argument(
        "--rebuild",
        action="store_true",
        help="delete the model file and make a new one",
    )
    args = argp.parse_args()

    if args.mania:
        modelmania()
        return

    model = None

    if args.rebuild:
        try:
            os.unlink(args.model_file)
        except FileNotFoundError:
            pass

    # create the model if it doesn't exist on disk
    if not os.path.isfile(args.model_file):
        model = train_model(args.model_file)

    # load the model
    if not model:
        model = load_model(args.model_file)

    # Test the model
    print(f"tested diff: {test_model(model)}")


if __name__ == "__main__":
    main()
