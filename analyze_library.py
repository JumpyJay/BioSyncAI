"""
Automated Audio Analysis for BioSyncAI Music Library
=====================================================
Scans music_library/{state}/ folders, extracts musical attributes
using librosa, and writes metadata.json per track.

Run once after populating music_library with MP3 files:
    python analyze_library.py

Attributes extracted:
    - tempo        : beats per minute (float)
    - brightness   : mean spectral centroid (0.0–1.0 normalised)
    - rhythmic_density : note onset rate relative to track length (0.0–1.0)
    - harmonic_complexity : unique pitch-class variety (0.0–1.0)
    - valence      : estimated emotional valence (negative / neutral / positive)
"""

import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

MUSIC_LIBRARY = "music_library"
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a")


def analyse_track(filepath: str) -> dict:
    """Extract musical attributes from a single audio file using librosa."""
    import librosa
    import scipy.signal as sp_signal

    # Load at 22.05 kHz for speed, keep mono
    # allow_press_error=False suppresses the flood of stderr messages from libmpg123
    # on corrupt MPEG headers; we handle decode failures gracefully instead
    try:
        y, sr = librosa.load(filepath, sr=22050, mono=True, duration=600)
    except Exception as exc:
        raise RuntimeError(f"audio decode failed ({exc})") from exc

    if y.size == 0:
        raise RuntimeError("decoded audio is empty")

    duration = float(librosa.get_duration(y=y, sr=sr))

    # --- Tempo (BPM) ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.ravel(tempo)[0])  # ensure scalar

    # --- Spectral Brightness (treble content) ---
    # Spectral centroid: high = bright, low = dark
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # Normalise centroid to roughly 0–1 using common range 0–8000 Hz
    brightness = float(np.clip(spectral_centroid.mean() / 8000, 0, 1))

    # --- Rhythmic Density (note onset rate) ---
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    onset_rate = len(onsets) / duration if duration > 0 else 0
    # Normalise: ~0.5 onsets/sec = sparse, ~4+ = dense
    rhythmic_density = float(np.clip(onset_rate / 4.0, 0, 1))

    # --- Harmonic Complexity (unique pitch-class variety) ---
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Fraction of distinct pitch classes used
    pitch_present = (chroma > 0.1).mean(axis=1)
    unique_pitch_ratio = np.sum(pitch_present > 0.1) / 12.0
    # Additional measure: how evenly are pitches distributed
    entropy = -np.sum(pitch_present * np.log(pitch_present + 1e-8))
    entropy_norm = entropy / np.log(12.0)  # 0 = single pitch, 1 = even
    harmonic_complexity = float(
        np.clip((unique_pitch_ratio * 0.5 + entropy_norm * 0.5), 0, 1)
    )

    # --- Valence (crude estimate from spectral features) ---
    # High brightness + relatively more high-frequency energy → positive;
    # low brightness + relatively more low-frequency energy → negative
    # Low-pass filter at 500 Hz to isolate low-frequency signal content
    nyq = sr / 2
    low_cutoff = min(500, nyq - 1)
    b, a = sp_signal.butter(4, low_cutoff / nyq, btype="low")
    y_low = sp_signal.filtfilt(b, a, y)

    energy_low = np.mean(y_low**2)       # actual low-freq signal energy
    energy_total = np.mean(y**2) + 1e-8  # total signal energy
    low_freq_ratio = energy_low / energy_total

    if brightness > 0.55 and low_freq_ratio < 0.4:
        valence = "positive"
    elif brightness < 0.35 and low_freq_ratio > 0.6:
        valence = "negative"
    else:
        valence = "neutral"

    return {
        "filename": os.path.basename(filepath),
        "duration_sec": round(duration, 1),
        "tempo": round(tempo, 1),
        "brightness": round(brightness, 3),
        "rhythmic_density": round(rhythmic_density, 3),
        "harmonic_complexity": round(harmonic_complexity, 3),
        "valence": valence,
    }


def scan_library():
    """Scan all audio files in music_library subfolders."""
    tracks = {}  # state -> list of track metadata

    for state in os.listdir(MUSIC_LIBRARY):
        state_path = os.path.join(MUSIC_LIBRARY, state)
        if not os.path.isdir(state_path):
            continue

        tracks[state] = []
        for fname in sorted(os.listdir(state_path)):
            if not fname.lower().endswith(AUDIO_EXTENSIONS):
                continue
            fpath = os.path.join(state_path, fname)
            print(f"  Analysing {state}/{fname}...", end=" ")
            try:
                meta = analyse_track(fpath)
                tracks[state].append(meta)
                print(
                    f"tempo={meta['tempo']:.0f}  "
                    f"brightness={meta['brightness']:.2f}  "
                    f"rhythmic_density={meta['rhythmic_density']:.2f}  "
                    f"harmonic_complexity={meta['harmonic_complexity']:.2f}  "
                    f"valence={meta['valence']}"
                )
            except Exception as e:
                print(f"  ✗ Error: {e}")

    return tracks


def write_metadata(tracks: dict):
    """Write metadata.json per state folder."""
    for state, track_list in tracks.items():
        meta_path = os.path.join(MUSIC_LIBRARY, state, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"state": state, "tracks": track_list}, f, indent=2)
        print(f"  ✓ Wrote {meta_path} ({len(track_list)} tracks)")


def main():
    print("=" * 60)
    print("BioSyncAI — Music Library Analyser")
    print("=" * 60)

    # Check librosa is available
    try:
        import librosa
    except ImportError:
        print(
            "ERROR: librosa not installed.\n"
            "  pip install librosa\n"
            "  (also run: pip install soundfile)"
        )
        return

    print(f"\nScanning: {os.path.abspath(MUSIC_LIBRARY)}")
    tracks = scan_library()
    print(f"\nTotal states: {len(tracks)}")
    for state, track_list in tracks.items():
        print(f"  {state}: {len(track_list)} tracks")

    print("\nWriting metadata files...")
    write_metadata(tracks)
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
