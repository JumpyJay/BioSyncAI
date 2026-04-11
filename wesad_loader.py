"""
WESAD Dataset Loader
====================
Parses and preprocesses a locally-downloaded copy of the WESAD dataset.

WESAD: Wearable Stress and Affect Detection
- Schmidt et al., 2018
- Download: https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset/data

Usage:
    python wesad_loader.py --wesad-path /path/to/WESAD

Prerequisites:
    Download and unzip WESAD to a local directory first. This script does NOT
    auto-download the dataset.

The script will:
    1. Walk the WESAD directory for S*/S*.pkl subject files
    2. Run Pan-Tompkins R-peak detection on the chest ECG (512 Hz)
    3. Compute sliding-window HRV features per 30 s window:
         hr_bpm, rmssd, rr_std
    4. Label each window as 'baseline' (neutral) or 'stress'
    5. Save processed data to wesad_processed.csv
"""

import os
import argparse
import pickle
import numpy as np
import scipy.signal as sp_signal


# -------------------------------------------------------------------
# Physiologically valid RR interval bounds (ms)
# -------------------------------------------------------------------
RR_MIN_MS = 300
RR_MAX_MS = 2000


# -------------------------------------------------------------------
# Helper: parse one subject's .pkl file
# -------------------------------------------------------------------
def load_subject_pkl(pkl_path):
    """Load a single subject's WESAD .pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def extract_session_features(data, window_sec=30, step_sec=15):
    """Sliding-window feature extraction over one session.

    Data keys (WESAD):
        signal.chest.ECG — ECG at 512 Hz
        label — per-sample labels at 700 Hz
        Both share the same number of samples; duration is in label-time.

    Labels:
        0 = not defined / ambiguous
        1 = baseline
        2 = stress
        3 = amusement
        4 = meditation
        6/7 = not used
    We use: baseline=neutral, stress=stress

    hr_delta: first-order difference in HR between consecutive windows
              (positive = increasing arousal, negative = decreasing)
    """
    chest = data["signal"]["chest"]
    labels = data["label"].ravel()  # 1D array, at ~700 Hz
    ecg = chest["ECG"][:, 0]

    fs_label = 700   # label sample rate
    fs_ecg = 512     # ECG sample rate
    winsize_labels = window_sec * fs_label       # 21000 for 30s
    step_labels = step_sec * fs_label           # 10500 for 15s step

    records = []
    prev_hr = None
    for start_lbl in range(0, len(labels) - winsize_labels, step_labels):
        # Dominant label in this window (mode)
        window_labels = labels[start_lbl : start_lbl + winsize_labels]
        values, counts = np.unique(window_labels, return_counts=True)
        lbl = int(values[np.argmax(counts)])

        if lbl == 0:
            continue  # skip undefined

        # Map label-space to ECG-space for peak detection
        start_ecg = int(start_lbl * fs_ecg / fs_label)
        end_ecg = int((start_lbl + winsize_labels) * fs_ecg / fs_label)
        ecg_window = ecg[start_ecg:end_ecg]

        # Detect peaks and compute HRV on this ECG window
        peaks = _detect_peaks_simple(ecg_window, fs_ecg)
        rr_ms = np.diff(peaks) / fs_ecg * 1000
        rr_clean = _filter_rr_outliers(rr_ms)

        if len(rr_clean) < 2:
            continue

        feat = compute_hrv_features_from_rr(rr_clean)
        if feat["hr_bpm"] == 0:
            continue

        # Compute HR delta across consecutive windows
        if prev_hr is not None:
            feat["hr_delta"] = feat["hr_bpm"] - prev_hr
        else:
            feat["hr_delta"] = 0.0
        prev_hr = feat["hr_bpm"]

        if lbl == 1:
            label = "baseline"
        elif lbl == 2:
            label = "stress"
        else:
            continue  # skip amusement / meditation for now

        records.append({**feat, "label": label})

    return records


def _detect_peaks_simple(signal, fs, min_distance_ms=300):
    """Simple R-peak detection on ECG signal (Pan-Tompkins inspired)."""
    min_dist = int(min_distance_ms / 1000 * fs)

    # Bandpass 5–15 Hz
    nyq = fs / 2
    b, a = sp_signal.butter(3, [5 / nyq, 15 / nyq], btype='band')
    ecg_filt = sp_signal.filtfilt(b, a, signal)

    # Differentiate and square
    diff = np.diff(ecg_filt)
    squared = np.concatenate([[0], diff]) ** 2

    # Moving average
    win = int(0.15 * fs)
    ma = np.convolve(squared, np.ones(win) / win, mode='same')

    # Find peaks
    threshold = np.percentile(ma, 60)
    peaks, _ = sp_signal.find_peaks(ma, distance=min_dist, height=threshold)
    return peaks


def _filter_rr_outliers(rr_intervals):
    """Reject physiologically impossible RR intervals."""
    if len(rr_intervals) == 0:
        return np.array([])
    valid = (rr_intervals >= RR_MIN_MS) & (rr_intervals <= RR_MAX_MS)
    rr_clean = rr_intervals[valid]
    if len(rr_clean) >= 3:
        kernel = np.ones(3) / 3
        rr_clean = np.convolve(rr_clean, kernel, mode='same')
    return rr_clean


def compute_hrv_features_from_rr(rr_intervals):
    """Compute HRV features from a clean RR interval array."""
    if len(rr_intervals) < 2:
        return {"hr_bpm": 0.0, "rmssd": 0.0, "rr_std": 0.0}

    hr_bpm = 60000.0 / np.mean(rr_intervals)
    successive_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))
    rr_std = np.std(rr_intervals)

    return {
        "hr_bpm": float(hr_bpm),
        "rmssd": float(rmssd),
        "rr_std": float(rr_std),
    }


def load_wesad(wesad_root):
    """Load all subjects from the WESAD directory.

    Expected structure:
        WESAD/
          S1/
            S1.pkl
            ...
          S2/
            S2.pkl
          ...

    Returns:
        List of dicts with keys: hr_bpm, rmssd, hr_delta, rr_std, label, subject
    """
    all_records = []

    subjects = [d for d in os.listdir(wesad_root) if d.startswith("S")]
    for subject in sorted(subjects):
        subject_dir = os.path.join(wesad_root, subject)
        pkl_files = [f for f in os.listdir(subject_dir) if f.endswith(".pkl")]
        if not pkl_files:
            continue

        pkl_path = os.path.join(subject_dir, pkl_files[0])
        print(f"  Processing {subject}/{pkl_files[0]}...", end=" ")
        try:
            data = load_subject_pkl(pkl_path)
        except Exception as e:
            print(f"✗ {e}")
            continue

        chest = data["signal"]["chest"]
        labels = data["label"]

        records = extract_session_features(data)
        for r in records:
            r["subject"] = subject
        all_records.extend(records)
        print(f"✓ {len(records)} windows")

    return all_records


def save_csv(records, output_path):
    """Save processed records to CSV."""
    import csv

    if not records:
        print("  No records to save.")
        return

    fieldnames = ["subject", "hr_bpm", "rmssd", "rr_std", "hr_delta", "label"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"  ✓ Saved {len(records)} records to {output_path}")


def main():
    print("=" * 60)
    print("WESAD Dataset Loader")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Load and preprocess WESAD dataset")
    parser.add_argument(
        "--wesad-path",
        type=str,
        required=True,
        help="Path to unzipped WESAD directory (containing S1/, S2/, .../)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="wesad_processed.csv",
        help="Output CSV path (default: wesad_processed.csv)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.wesad_path):
        print(
            f"ERROR: WESAD directory not found: {args.wesad_path}\n"
            "  1. Download from https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset/data\n"
            "  2. Download WESAD.zip\n"
            "  3. Unzip to a directory and pass --wesad-path to this script"
        )
        return

    print(f"\nLoading WESAD from: {os.path.abspath(args.wesad_path)}")
    records = load_wesad(args.wesad_path)

    if not records:
        print("ERROR: No records extracted. Check WESAD directory structure.")
        return

    # Label distribution
    from collections import Counter
    label_counts = Counter(r["label"] for r in records)
    print(f"\nLabel distribution: {dict(label_counts)}")

    save_csv(records, args.output)
    print(f"\n✓ WESAD preprocessing complete!")
    print(f"  Next: python train_svm.py --data {args.output}")


if __name__ == "__main__":
    main()
