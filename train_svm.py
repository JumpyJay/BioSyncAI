"""
WESAD SVM Training Script
=========================
Loads processed WESAD data, trains an SVM classifier (binary: baseline vs. stress),
evaluates with train/test split, and saves the model to svm_wesad.pkl.

Usage:
    python train_svm.py --data wesad_processed.csv

Outputs:
    svm_wesad.pkl  — trained SVM model + scaler
    classification_report.txt  — precision/recall/F1 per class
"""

import argparse
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


LABEL_ENCODER = LabelEncoder()


def load_data(csv_path):
    """Load processed WESAD CSV."""
    try:
        import csv
    except ImportError:
        import csv

    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def prepare_features(records):
    """Extract feature matrix X and label vector y from records.

    Features: hr_bpm, rmssd, hr_delta, rr_std
    Binary labels: baseline (neutral) vs. stress
    """
    X = []
    y = []

    for r in records:
        # Skip non-binary labels (if meditation/amusement present)
        if r["label"] not in ("baseline", "stress"):
            continue
        try:
            x = [
                float(r["hr_bpm"]),
                float(r["rmssd"]),
                float(r["hr_delta"]),
                float(r["rr_std"]),
            ]
            X.append(x)
            y.append(r["label"])
        except (ValueError, KeyError):
            continue

    X = np.array(X)
    y = LABEL_ENCODER.fit_transform(y)  # baseline=0, stress=1
    return X, y


def train_svm(X_train, y_train, kernel="rbf"):
    """Train an SVM with StandardScaler + RBF kernel."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, probability=True, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(pipeline, X_test, y_test, label_names):
    """Print classification report and confusion matrix."""
    y_pred = pipeline.predict(X_test)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (WESAD — baseline vs. stress)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=label_names))

    print("CONFUSION MATRIX")
    cm = confusion_matrix(y_test, y_pred)
    print(f"               Pred baseline  Pred stress")
    print(f"Actual baseline     {cm[0][0]:4d}         {cm[0][1]:4d}")
    print(f"Actual stress       {cm[1][0]:4d}         {cm[1][1]:4d}")

    accuracy = (y_pred == y_test).mean()
    print(f"\nOverall accuracy: {accuracy:.1%}")


def save_model(pipeline, output_path="svm_wesad.pkl"):
    """Save trained pipeline (scaler + SVM) to disk."""
    with open(output_path, "wb") as f:
        pickle.dump({
            "pipeline": pipeline,
            "label_encoder": LABEL_ENCODER,
        }, f)
    print(f"\n✓ Model saved to {output_path}")


def compare_kernels(X_train, X_test, y_train, y_test, label_names):
    """Compare linear vs RBF kernel and report."""
    print("\n--- Kernel Comparison ---")
    for kernel in ("linear", "rbf"):
        pipeline = train_svm(X_train, y_train, kernel=kernel)
        acc = (pipeline.predict(X_test) == y_test).mean()
        print(f"  {kernel:8s}: test accuracy = {acc:.1%}")


def main():
    print("=" * 60)
    print("WESAD SVM Training — Binary: baseline vs. stress")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Train SVM on WESAD data")
    parser.add_argument(
        "--data",
        type=str,
        default="wesad_processed.csv",
        help="Path to processed WESAD CSV from wesad_loader.py",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test split (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="svm_wesad.pkl",
        help="Output model path (default: svm_wesad.pkl)",
    )
    args = parser.parse_args()

    if not __import__("os").path.exists(args.data):
        print(f"ERROR: CSV not found: {args.data}")
        print("  Run: python wesad_loader.py --wesad-path /path/to/WESAD")
        return

    print(f"\nLoading data from {args.data}...")
    records = load_data(args.data)
    print(f"  Loaded {len(records)} records")

    X, y = prepare_features(records)
    label_names = list(LABEL_ENCODER.classes_)
    print(f"  Features shape: {X.shape}")
    print(f"  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Label mapping: baseline=0, stress=1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(y_train)} | Test: {len(y_test)}")

    # Kernel comparison
    compare_kernels(X_train, X_test, y_train, y_test, label_names)

    # Train final model with RBF
    print("\nTraining final SVM (RBF kernel)...")
    pipeline = train_svm(X_train, y_train, kernel="rbf")

    evaluate(pipeline, X_test, y_test, label_names)
    save_model(pipeline, args.output)

    print("\n✓ Training complete!")
    print("  Next: load svm_wesad.pkl in biofeedback_system.py for pre-trained inference")


if __name__ == "__main__":
    main()
