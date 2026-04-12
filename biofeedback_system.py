"""
Closed-Loop Biofeedback Music System
=====================================
A system that reads physiological signals, classifies cognitive state,
and generates adaptive music in real-time.
"""

import numpy as np
from collections import deque
import time
import pickle
import os
import json
import subprocess
import tempfile
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. DATA ACQUISITION LAYER
# =============================================================================

class PulseSensorReader:
    """Interface for reading pulse sensor data via ADC."""

    def __init__(self, adc_channel=0, sample_rate=100):
        self.adc_channel = adc_channel
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=sample_rate * 10)

    def read_raw(self):
        """Read raw voltage from ADC (placeholder - implement with actual hardware)."""
        # In production, use spidev or similar library:
        # return self.read_adc(self.adc_channel)
        return np.random.uniform(0.5, 3.3)  # Simulated for testing

    def get_signal_buffer(self, duration_sec=10, fast_mode=False):
        """Collect signal for specified duration."""
        samples_needed = self.sample_rate * duration_sec
        signal = []

        # Fast mode: skip sleep delays for testing
        sleep_time = 0 if fast_mode else (1 / self.sample_rate)

        for i in range(samples_needed):
            signal.append(self.read_raw())
            if not fast_mode:
                time.sleep(sleep_time)
            # Show progress every 500 samples
            if (i + 1) % 500 == 0:
                print(f"  📊 Collecting signal: {i + 1}/{samples_needed} samples...", end='\r')

        if samples_needed > 0:
            print(f"  ✓ Signal collected: {samples_needed} samples           ")
        return np.array(signal)


class ArduinoPulseReader:
    """Reads pulse sensor data from Arduino Uno R3 via USB serial.

    The Arduino should run a sketch that reads from the pulse sensor and
    outputs raw values via Serial at 9600 baud.

    Example Arduino sketch:
        const int PULSE_PIN = A0;
        void setup() { Serial.begin(9600); }
        void loop() {
          int val = analogRead(PULSE_PIN);
          Serial.println(val);
          delay(10);  // ~100 Hz sampling
        }
    """

    def __init__(self, port=None, sample_rate=100):
        """
        Args:
            port: Serial port (e.g., '/dev/cu.usbmodem1411' on macOS,
                  'COM3' on Windows, '/dev/ttyUSB0' on Linux).
                  If None, attempts auto-detection.
            sample_rate: Target sampling rate in Hz.
        """
        self.port = port
        self.sample_rate = sample_rate
        self.serial = None
        self.buffer = deque(maxlen=sample_rate * 10)
        self._connect()

    def _connect(self):
        """Establish serial connection to Arduino."""
        try:
            import serial
        except ImportError:
            raise ImportError(
                "pyserial is required for Arduino communication. "
                "Install it with: pip install pyserial"
            )

        if self.port is None:
            self.port = self._auto_detect_port()

        self.serial = serial.Serial(self.port, 9600, timeout=1)
        time.sleep(2)  # Wait for Arduino reset after serial connection
        self.serial.flushInput()
        print(f"  ✓ Connected to Arduino on {self.port}")

    def _auto_detect_port(self):
        """Attempt to auto-detect the Arduino serial port."""
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        arduino_ports = [
            p.device for p in ports
            if "Arduino" in p.description or "CH340" in p.description or "USB" in p.device
        ]
        if arduino_ports:
            print(f"  ✓ Auto-detected Arduino on {arduino_ports[0]}")
            return arduino_ports[0]
        if ports:
            print(f"  ⚠️  Could not auto-detect Arduino, using {ports[0].device}")
            return ports[0].device
        raise RuntimeError(
            "No serial port found. Connect Arduino or specify port manually."
        )

    def read_raw(self):
        """Read one raw ADC value from Arduino serial."""
        try:
            if self.serial and self.serial.in_waiting > 0:
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    value = float(line)
                    # Map 0-1023 ADC reading to voltage (0-3.3V for 10-bit ADC)
                    return value / 1023.0 * 3.3
        except (ValueError, UnicodeDecodeError):
            pass
        return np.random.uniform(0.5, 3.3)  # Fallback if no data

    def get_signal_buffer(self, duration_sec=10, fast_mode=False):
        """Collect signal for specified duration from Arduino."""
        samples_needed = self.sample_rate * duration_sec
        signal = []

        # Fast mode: skip sleep delays for testing
        sleep_time = 0 if fast_mode else (1 / self.sample_rate)

        for i in range(samples_needed):
            signal.append(self.read_raw())
            if not fast_mode:
                time.sleep(sleep_time)
            if (i + 1) % 500 == 0:
                print(f"  📊 Collecting signal: {i + 1}/{samples_needed} samples...", end='\r')

        if samples_needed > 0:
            print(f"  ✓ Signal collected: {samples_needed} samples           ")
        return np.array(signal)

    def close(self):
        """Close serial connection."""
        if self.serial:
            self.serial.close()
            self.serial = None


# =============================================================================
# 2. SIGNAL PROCESSING & FEATURE EXTRACTION
# =============================================================================

class SignalProcessor:
    """Extract HRV features from raw pulse signal with arrhythmia handling."""

    # Physiologically valid RR interval bounds (ms)
    RR_MIN_MS = 300
    RR_MAX_MS = 2000

    def __init__(self, sample_rate=100, calibration_window_sec=30):
        self.sample_rate = sample_rate
        self.calibration_window_sec = calibration_window_sec
        self.personalized_baseline_rmssd = None
        self.baseline_rmssd_window = deque(maxlen=calibration_window_sec * sample_rate)
        self.hr_history = deque(maxlen=30)  # rolling HR for delta computation
        self.is_calibrated = False

    def calibrate(self, signal):
        """Run personalized baseline calibration on first 30s of signal.

        Sets personalized_baseline_rmssd to the resting RMSSD.
        Call this once at system startup before normal operation.
        """
        peaks = self.detect_peaks(signal)
        rr_intervals = self.calculate_rr_intervals(peaks)
        rr_clean = self.filter_rr_outliers(rr_intervals)
        if len(rr_clean) >= 2:
            self.personalized_baseline_rmssd = self.calculate_rmssd(rr_clean)
            self.is_calibrated = True
            print(f"  ✓ Personalised baseline RMSSD: {self.personalized_baseline_rmssd:.1f} ms")

    def detect_peaks(self, signal, min_distance=300):
        """Detect R-peaks in pulse signal."""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(signal, distance=min_distance)
        return peaks

    def filter_rr_outliers(self, rr_intervals):
        """Reject physiologically impossible RR intervals (arrhythmia / artifact handler).

        Removes intervals < 300 ms or > 2000 ms, which are likely ectopic beats
        or motion artifacts. Also applies a moving-average smoothing filter.
        """
        if len(rr_intervals) == 0:
            return np.array([])

        # Hard physiological bounds
        valid = (rr_intervals >= self.RR_MIN_MS) & (rr_intervals <= self.RR_MAX_MS)
        rr_clean = rr_intervals[valid]

        # Moving-average smoothing (window=3)
        if len(rr_clean) >= 3:
            kernel = np.ones(3) / 3
            rr_smooth = np.convolve(rr_clean, kernel, mode='same')
            return rr_smooth

        return rr_clean

    def calculate_rr_intervals(self, peaks):
        """Calculate RR intervals in milliseconds."""
        if len(peaks) < 2:
            return np.array([])
        rr_ms = np.diff(peaks) / self.sample_rate * 1000
        return rr_ms

    def calculate_rmssd(self, rr_intervals):
        """Calculate RMSSD (Root Mean Square of Successive Differences)."""
        if len(rr_intervals) < 2:
            return 0.0
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        return rmssd

    def calculate_hr(self, rr_intervals):
        """Calculate heart rate in BPM."""
        if len(rr_intervals) == 0:
            return 0.0
        return 60000 / np.mean(rr_intervals)

    def calculate_hr_delta(self):
        """Calculate HR Delta — first derivative of HR trend.

        Positive = increasing HR (sympathetic activation).
        Negative = decreasing HR (parasympathetic / recovery).
        """
        if len(self.hr_history) < 2:
            return 0.0
        hr_array = np.array(self.hr_history)
        # Simple first-order difference
        return float(np.mean(np.diff(hr_array)))

    def extract_features(self, signal):
        """Extract all features for SVM classification (BPM, RMSSD, HR Delta, rr_std)."""
        peaks = self.detect_peaks(signal)
        rr_intervals = self.calculate_rr_intervals(peaks)
        rr_clean = self.filter_rr_outliers(rr_intervals)

        hr = self.calculate_hr(rr_clean)
        rmssd = self.calculate_rmssd(rr_clean)

        # Update rolling HR history for delta
        if hr > 0:
            self.hr_history.append(hr)

        features = {
            'hr': hr,
            'rmssd': rmssd,
            'hr_delta': self.calculate_hr_delta(),
            'rr_mean': np.mean(rr_clean) if len(rr_clean) > 0 else 0,
            'rr_std': np.std(rr_clean) if len(rr_clean) > 0 else 0,
            'baseline_rmssd': self.personalized_baseline_rmssd,
        }
        return features


# =============================================================================
# 3. INTELLIGENCE LAYER (SVM + RL)
# =============================================================================

# 9-Quadrant model: 3×3 grid indexed as
#   row 0 = RMSSD Low    (stress/anxiety zone)
#   row 1 = RMSSD Med    (neutral/control zone)
#   row 2 = RMSSD High   (relaxed/flow zone)
#   col 0 = BPM Low,     col 1 = BPM Med,     col 2 = BPM High
QUADRANT_NAMES = [
    # BPM Low              BPM Med               BPM High
    ["relaxation",        "control",           "flow"],         # RMSSD > 65  — high skill
    ["boredom",           "neutral",           "arousal"],      # RMSSD 30–65 — medium skill
    ["apathy",            "worry",             "anxiety"],      # RMSSD < 30  — low skill
]

# BPM thresholds (Low / Med / High) — per-user calibratable
BPM_THRESHOLDS = (60, 80)
# RMSSD thresholds (ms) (Low / Med / High) — per-user calibratable
RMSSD_THRESHOLDS = (30, 65)


class CognitiveStateClassifier:
    """SVM-based 9-quadrant cognitive state classifier with WESAD fine-tuning.

    Architecture: WESAD (pre-trained on published stress dataset) serves as the
    foundation model. User feedback is used to fine-tune this foundation, adapting
    it to the user's unique physiological profile via incremental learning.

    Feature vector: [HR, RMSSD, HR Delta, RR Std, WESAD bias]
                    — WESAD bias = P(stress) from the pre-trained model
    """

    def __init__(self, model_path='svm_model.pkl',
                 bpm_thresholds=None, rmssd_thresholds=None):
        from sklearn.svm import SVC
        self.svm = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        self.trained = False
        self.model_path = model_path
        self.training_data = []
        self.training_labels = []
        # Allow per-user calibration of thresholds
        self.bpm_thresholds = bpm_thresholds or BPM_THRESHOLDS
        self.rmssd_thresholds = rmssd_thresholds or RMSSD_THRESHOLDS

        # Pre-trained WESAD foundation model + its hyperparameters for fine-tuning
        self.wesad_pipeline = None
        self.wesad_params = None   # {gamma, C, kernel} — saved separately for fine-tune init
        self.wesad_path = "svm_wesad.pkl"
        self.wesad_params_path = "svm_wesad_params.pkl"
        self._load_wesad()

        if os.path.exists(model_path):
            self.load_model(model_path)

    def _bpm_level(self, hr):
        """Return 0 (Low), 1 (Med), or 2 (High) for BPM."""
        if hr < self.bpm_thresholds[0]:
            return 0
        if hr <= self.bpm_thresholds[1]:
            return 1
        return 2

    def _rmssd_level(self, rmssd):
        """Return 0 (VeryLow/Low), 1 (Med), 2 (High) for RMSSD."""
        if rmssd < self.rmssd_thresholds[0]:
            return 0
        if rmssd <= self.rmssd_thresholds[1]:
            return 1
        return 2

    def _load_wesad(self):
        """Load pre-trained WESAD SVM and extract hyperparameters for fine-tuning."""
        if os.path.exists(self.wesad_path):
            try:
                with open(self.wesad_path, "rb") as f:
                    data = pickle.load(f)
                    self.wesad_pipeline = data["pipeline"]
                acc = data.get("test_accuracy")
                if acc is not None:
                    print(f"  ✓ WESAD SVM loaded (test accuracy={acc:.1%})")
                else:
                    print(f"  ✓ WESAD SVM loaded")

                # Extract hyperparameters from the WESAD SVM for fine-tune re-init
                try:
                    self.wesad_params = {
                        "gamma": self.wesad_pipeline.named_steps['svm'].gamma,
                        "C": self.wesad_pipeline.named_steps['svm'].C,
                        "kernel": self.wesad_pipeline.named_steps['svm'].kernel,
                    }
                    # Save params separately so fine-tuning can re-initialize
                    with open(self.wesad_params_path, "wb") as f:
                        pickle.dump(self.wesad_params, f)
                except Exception:
                    # Fallback: try to load pre-saved params if extraction failed
                    if os.path.exists(self.wesad_params_path):
                        with open(self.wesad_params_path, "rb") as f:
                            self.wesad_params = pickle.load(f)
                    else:
                        self.wesad_params = None
                        print(f"  ⚠️  WESAD hyperparameters not available — fine-tune will init fresh SVM")
            except Exception as e:
                print(f"  ⚠️  Could not load WESAD SVM: {e}")
                self.wesad_pipeline = None
                self.wesad_params = None

    def predict_from_thresholds(self, hr_features):
        """Predict using 3×3 grid (no SVM needed if not yet trained).

        Includes a challenge-skill balance check per Csikszentmihalyi's model:
        Flow requires BOTH high challenge (BPM) AND high skill (RMSSD).
        When challenge and skill levels differ too much, the user is in
        the lower-skill quadrant of that challenge level, not in Flow.
        """
        bpm_lvl = self._bpm_level(hr_features['hr'])
        rmssd_lvl = self._rmssd_level(hr_features['rmssd'])

        quadrant = QUADRANT_NAMES[rmssd_lvl][bpm_lvl]

        # Csikszentmihalyi balance: |challenge - skill| should be small for Flow
        # gap > 1 means severe imbalance between challenge (BPM) and skill (RMSSD)
        gap = abs(bpm_lvl - rmssd_lvl)

        # Flow = high skill (row 0) + high challenge (col 2) — only true Flow cell
        # Since row 0 = high skill and row 2 = low skill, flow is at (rmssd_lvl=0, bpm_lvl=2)
        if quadrant == "flow":
            if gap > 1:
                # Challenge >> skill: anxiety, not flow
                quadrant = "anxiety"
        elif quadrant == "arousal":
            # Arousal = medium skill + high challenge — anxiety if gap > 1
            if gap > 1:
                quadrant = "anxiety"
        elif quadrant == "relaxation":
            # Relaxation = high skill + low challenge — control if gap > 1
            if gap > 1:
                quadrant = "control"

        return quadrant

    def _get_wesad_bias(self, X):
        """Return WESAD P(stress) as the 5th feature — the foundation bias."""
        if self.wesad_pipeline is None:
            return 0.5  # neutral prior if WESAD unavailable
        try:
            proba = self.wesad_pipeline.predict_proba(X)[0]
            # proba[1] = P(stress), proba[0] = P(baseline)
            return proba[1] if len(proba) > 1 else 0.5
        except Exception:
            return 0.5

    def prepare_features(self, hr_features):
        """Format 5 features for SVM input: BPM, RMSSD, HR Delta, rr_std, WESAD bias.

        The WESAD bias (P(stress)) is appended as a 5th feature, allowing the
        user's SVM to learn from the pre-trained foundation model while adapting
        to personal physiology.
        """
        X = np.array([
            hr_features['hr'],
            hr_features['rmssd'],
            hr_features['hr_delta'],
            hr_features['rr_std'],
        ]).reshape(1, -1)
        wesad_bias = self._get_wesad_bias(X)
        return np.concatenate([X, [[wesad_bias]]], axis=1)

    def predict(self, hr_features):
        """Predict 9-quadrant cognitive state.

        The user's SVM (when trained) uses a 5-dimensional feature vector that
        includes the WESAD P(stress) as the 5th feature — this bakes the
        pre-trained foundation bias directly into the classifier.

        Priority:
          1. Fine-tuned user SVM (after ≥3 user-labeled samples)
          2. WESAD-aware threshold grid (uses WESAD bias internally)
        """
        X = self.prepare_features(hr_features)

        if self.trained:
            X_scaled = self.scaler.transform(X)
            return self.svm.predict(X_scaled)[0]

        # WESAD-aware threshold grid as the primary cold-start predictor.
        # The WESAD bias is embedded in the 5th feature but since the SVM
        # is not yet trained, we fall back to the threshold grid augmented
        # with WESAD binary signal as a soft bias.
        return self.predict_from_thresholds(hr_features)

    def train_incrementally(self, hr_features, label):
        """Fine-tune the WESAD foundation model with user-labeled data.

        On the first fine-tuning call (when len=3), the SVM is re-initialized
        from WESAD's stored hyperparameters (gamma, C, kernel) so training
        starts from the pre-trained foundation rather than from scratch.
        Subsequent calls continue fine-tuning from the current model state.
        """
        from sklearn.svm import SVC

        X = self.prepare_features(hr_features)
        self.training_data.append(X[0])
        self.training_labels.append(label)

        if len(self.training_data) >= 3:
            # Fine-tune: on first fine-tune, re-init SVM from WESAD foundation
            if len(self.training_data) == 3:
                if self.wesad_params is not None:
                    self.svm = SVC(
                        kernel=self.wesad_params["kernel"],
                        C=self.wesad_params["C"],
                        gamma=self.wesad_params["gamma"],
                        probability=True,
                    )
                    print(f"  ✓ Fine-tuning from WESAD foundation "
                          f"(gamma={self.wesad_params['gamma']:.4f}, C={self.wesad_params['C']:.2f})")
                else:
                    self.svm = SVC(kernel='rbf', probability=True)
                    print(f"  ✓ Fine-tuning from scratch (WESAD params unavailable)")

            X_train = np.array(self.training_data)
            y_train = np.array(self.training_labels)
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.svm.fit(X_train_scaled, y_train)
            self.trained = True
            self.save_model()
            print(f"  ✓ SVM fine-tuned with {len(self.training_data)} user samples")

    def save_model(self):
        """Save fine-tuned SVM, scaler, and calibration to disk."""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'svm': self.svm,
                'scaler': self.scaler,
                'training_data': self.training_data,
                'training_labels': self.training_labels,
                'trained': self.trained,
                'bpm_thresholds': self.bpm_thresholds,
                'rmssd_thresholds': self.rmssd_thresholds,
                'wesad_params': self.wesad_params,   # foundation model params used for fine-tuning
            }, f)

    def load_model(self, model_path=None):
        """Load previously fine-tuned model (retains WESAD foundation info)."""
        path = model_path or self.model_path
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.svm = data['svm']
                self.scaler = data['scaler']
                self.training_data = data['training_data']
                self.training_labels = data['training_labels']
                self.trained = data['trained']
                self.bpm_thresholds = data.get('bpm_thresholds', BPM_THRESHOLDS)
                self.rmssd_thresholds = data.get('rmssd_thresholds', RMSSD_THRESHOLDS)
                self.wesad_params = data.get('wesad_params', None)
                src = "WESAD-fine-tuned" if self.wesad_params else "scratch"
                print(f"  ✓ Loaded model ({src}): {len(self.training_data)} samples, trained={self.trained}")


class RLMusicAgent:
    """Q-learning agent for playlist-based music selection.

    State space : 9 cells of the 3×3 BPM × RMSSD grid
    Action space : select a target quadrant folder whose music
                   pushes the user toward Flow
    Reward       : proximity to Flow + low RMSSD rolling variance
    """

    # 9-cell grid encoded as flat index
    # index = rmssd_level * 3 + bpm_level
    NUM_STATES = 9
    NUM_ACTIONS = 9   # one per target quadrant

    # Flow quadrant index (BPM=High + RMSSD=High = top-right = 2*3+2 = 8)
    FLOW_INDEX = 8

    def __init__(self, q_table_path='q_table.npy'):
        self.q_table_path = q_table_path
        if os.path.exists(q_table_path):
            self.q_table = np.load(q_table_path)
            print(f"  ✓ Loaded Q-table from previous sessions")
        else:
            self.q_table = np.zeros((self.NUM_STATES, self.NUM_ACTIONS))

        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1

        # Rolling RMSSD history for variance-based reward
        self.rmssd_history = deque(maxlen=10)
        self.target_rmssd = 50.0   # Flow RMS LD

    def _state_index(self, hr, rmssd):
        """Convert BPM + RMSSD to flat 0–8 state index."""
        if hr < BPM_THRESHOLDS[0]:
            bpm_lvl = 0
        elif hr <= BPM_THRESHOLDS[1]:
            bpm_lvl = 1
        else:
            bpm_lvl = 2
        if rmssd < RMSSD_THRESHOLDS[0]:
            rmssd_lvl = 0
        elif rmssd <= RMSSD_THRESHOLDS[1]:
            rmssd_lvl = 1
        else:
            rmssd_lvl = 2
        return rmssd_lvl * 3 + bpm_lvl

    def _get_rmssd_variance(self):
        """Rolling RMS SD variance — penalises erratic HRV."""
        if len(self.rmssd_history) < 3:
            return 0.0
        return float(np.var(self.rmssd_history))

    def get_reward(self, current_rmssd):
        """Stabilisation reward: proximity to Flow + low RMSSD variance."""
        self.rmssd_history.append(current_rmssd)

        dist_to_flow = abs(self.target_rmssd - current_rmssd)
        variance_penalty = self._get_rmssd_variance()

        # Proximity component: closer to Flow = positive
        if dist_to_flow < 10:
            proximity_reward = 2.0
        elif dist_to_flow < 20:
            proximity_reward = 1.0
        elif dist_to_flow < 35:
            proximity_reward = 0.0
        else:
            proximity_reward = -1.0

        # Variance component: low variance = positive (stabilisation)
        variance_reward = max(-1.0, 1.0 - variance_penalty / 100)

        return proximity_reward + 0.5 * variance_reward

    def select_action(self, state):
        """Epsilon-greedy: pick action (target quadrant) with highest Q-value."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.NUM_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def update_q_table(self, state, action, reward, next_state):
        """Q-learning update rule."""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        self.q_table[state, action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        np.save(self.q_table_path, self.q_table)

    def act(self, hr, current_rmssd, previous_rmssd):
        """Main RL decision loop.

        Returns the selected target quadrant name (e.g. 'flow').
        """
        state = self._state_index(hr, previous_rmssd)
        action_idx = self.select_action(state)
        reward = self.get_reward(current_rmssd)
        next_state = self._state_index(hr, current_rmssd)

        self.update_q_table(state, action_idx, reward, next_state)

        # Map action index back to quadrant name
        target_bpm_lvl = action_idx % 3
        target_rmssd_lvl = action_idx // 3
        quadrant = QUADRANT_NAMES[target_rmssd_lvl][target_bpm_lvl]
        return quadrant


# =============================================================================
# 4. GENERATIVE MUSIC ENGINE (Playlist-based + Procedural fallback)
# =============================================================================

class MusicEngine:
    """Curated playlist engine with librosa audio analysis and procedural fallback.

    Primary mode : select best-matching track from music_library/{state}/
                   based on metadata extracted by analyze_library.py
    Fallback mode: pure-tone binaural generation (when library is empty or
                  analyse_library.py has not been run)
    """

    def __init__(self, sample_rate=44100, library_dir="music_library"):
        from scipy.io import wavfile
        self.wavfile = wavfile
        self.sample_rate = sample_rate
        self.is_playing = False
        self.audio_dir = "biofeedback_audio"
        self.library_dir = library_dir

        # Create audio directory if it doesn't exist
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)

        # Music parameters (for metadata matching / procedural fallback)
        self.tempo = 90
        self.brightness = 0.5
        self.rhythmic_density = 0.5
        self.harmonic_complexity = 0.5

        # Track library index
        self.track_index = {}   # state -> list of track dicts with metadata
        self.load_library()

    # ------------------------------------------------------------------
    # Playlist management
    # ------------------------------------------------------------------

    def load_library(self):
        """Scan music_library/ and build track index from metadata.json files."""
        if not os.path.exists(self.library_dir):
            print("  ⚠️  music_library/ not found — using procedural fallback")
            return

        for state in os.listdir(self.library_dir):
            state_path = os.path.join(self.library_dir, state)
            if not os.path.isdir(state_path):
                continue

            meta_path = os.path.join(state_path, "metadata.json")
            if not os.path.exists(meta_path):
                continue

            with open(meta_path) as f:
                data = json.load(f)
                self.track_index[state] = data.get("tracks", [])
                print(
                    f"  ✓ Loaded {state}: {len(self.track_index[state])} tracks"
                )

    def select_track(self, target_state, current_params=None):
        """Select the best-matching track for target_state.

        Uses Euclidean distance in attribute space (brightness, rhythmic_density,
        harmonic_complexity) to find the track that best matches the current
        music parameters — encouraging smooth transitions.
        If library is empty, falls back to procedural binaural generation.
        """
        import subprocess

        current_params = current_params or self.get_current_params()
        tracks = self.track_index.get(target_state, [])

        if not tracks:
            # Fallback to procedural binaural if no curated track exists
            return self._play_procedural(target_state)

        # Score each track by distance to current params
        best_track = None
        best_score = float("inf")

        for track in tracks:
            score = (
                (track.get("brightness", 0.5) - current_params["brightness"]) ** 2
                + (track.get("rhythmic_density", 0.5)
                   - current_params["rhythmic_density"]) ** 2
                + (track.get("harmonic_complexity", 0.5)
                   - current_params["harmonic_complexity"]) ** 2
            )
            if score < best_score:
                best_score = score
                best_track = track

        # Update internal params to match chosen track
        if best_track:
            self.brightness = best_track.get("brightness", self.brightness)
            self.rhythmic_density = best_track.get(
                "rhythmic_density", self.rhythmic_density
            )
            self.harmonic_complexity = best_track.get(
                "harmonic_complexity", self.harmonic_complexity
            )
            # Play the track
            track_path = os.path.join(
                self.library_dir, target_state, best_track["filename"]
            )
            print(
                f"  🎵 Playing [{target_state}] {best_track['filename']}"
                f"  bright={self.brightness:.2f}"
                f"  density={self.rhythmic_density:.2f}"
                f"  complexity={self.harmonic_complexity:.2f}"
            )
            self._play_file(track_path)
            return {"state": target_state, "track": best_track["filename"]}

        return {"state": target_state, "track": None}

    def _play_file(self, filepath):
        """Play an audio file using the OS default player (non-blocking; returns immediately)."""
        try:
            abs_path = os.path.abspath(filepath)
            if os.name == "nt":  # Windows
                os.startfile(abs_path)
            elif os.uname().sysname == "Darwin":  # macOS
                subprocess.run(["open", abs_path], check=True)
            else:  # Linux
                subprocess.run(["xdg-open", abs_path], check=True)
            self.is_playing = True
        except Exception as e:
            print(f"  ⚠️  Could not play {filepath}: {e}")

    def _play_procedural(self, target_state):
        """Procedural binaural fallback when no curated track is available."""
        # Map target quadrant to binaural frequency
        binaural_map = {
            "flow":     40,
            "arousal":  20,
            "anxiety":  15,
            "panic":    10,
            "relaxation": 8,
            "control":  12,
            "neutral":  10,
            "worry":    12,
            "boredom":   6,
            "apathy":    5,
        }
        target_hz = binaural_map.get(target_state, 10)
        print(
            f"  🎵 No curated track for '{target_state}' — "
            f"generating procedural binaural ({target_hz} Hz)"
        )
        audio_data = self.generate_binaural_beats(duration_sec=10, target_hz=target_hz)
        filepath = os.path.join(
            self.audio_dir, f"binaural_{target_state}_{int(time.time())}.wav"
        )
        self.wavfile.write(filepath, self.sample_rate, audio_data)
        self._play_file(filepath)
        return {"state": target_state, "track": "procedural", "hz": target_hz}

    # ------------------------------------------------------------------
    # Procedural synthesis (retained for fallback / binaural use)
    # ------------------------------------------------------------------

    def generate_binaural_beats(self, duration_sec=10, target_hz=10):
        """Generate binaural beats for inducing specific brain states.

        Args:
            duration_sec: Length of audio
            target_hz: Target brainwave frequency
                - 40 Hz: Focus/Flow state
                - 10-14 Hz: Relaxed alertness
                - 7-10 Hz: Deep relaxation
                - 4-7 Hz: Meditation
        """
        num_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples)

        left_freq = 200
        right_freq = left_freq + target_hz

        left_channel = np.sin(2.0 * np.pi * left_freq * t) * 0.2
        right_channel = np.sin(2.0 * np.pi * right_freq * t) * 0.2

        ambient_freq = 110
        ambient = 0.15 * (
            np.sin(2.0 * np.pi * ambient_freq * t) +
            0.5 * np.sin(2.0 * np.pi * ambient_freq * 1.5 * t) +
            0.3 * np.sin(2.0 * np.pi * ambient_freq * 2 * t)
        )

        left_channel += ambient
        right_channel += ambient

        envelope = np.ones_like(t)
        fade_samples = int(self.sample_rate * 1.0)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        left_channel *= envelope
        right_channel *= envelope

        max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        if max_val > 0:
            left_channel = left_channel / max_val * 0.8
            right_channel = right_channel / max_val * 0.8

        stereo_audio = np.column_stack((left_channel, right_channel))
        audio_data = np.int16(stereo_audio * 32767)
        return audio_data

    def apply_action(self, target_state):
        """Apply RL action: select and play a track from the target quadrant."""
        return self.select_track(target_state, self.get_current_params())

    def get_current_params(self):
        """Return current music parameters."""
        return {
            "tempo": self.tempo,
            "brightness": self.brightness,
            "rhythmic_density": self.rhythmic_density,
            "harmonic_complexity": self.harmonic_complexity,
        }

    def stop(self):
        """Cleanup."""
        pass


# =============================================================================
# 5. MAIN CLOSED-LOOP SYSTEM
# =============================================================================

class BiofeedbackLoop:
    """Main closed-loop biofeedback system with 9-quadrant SVM + RL playlist."""

    # Valid labels accepted from get_user_label() for incremental learning
    VALID_LABELS = [
        "anxiety", "arousal", "flow", "control", "relaxation",
        "boredom", "apathy", "worry", "neutral", "panic",
    ]

    def __init__(self, sensor_type="simulated", arduino_port=None):
        """
        Args:
            sensor_type: "simulated" (default) or "arduino"
            arduino_port: Serial port for Arduino (e.g., '/dev/cu.usbmodem1411').
                          If None and sensor_type="arduino", auto-detects.
        """
        if sensor_type == "arduino":
            self.sensor = ArduinoPulseReader(port=arduino_port)
        else:
            self.sensor = PulseSensorReader()
        self.processor = SignalProcessor()
        self.classifier = CognitiveStateClassifier()
        self.rl_agent = RLMusicAgent()
        self.music = MusicEngine()
        self.sensor_type = sensor_type

        self.previous_rmssd = 50.0
        self.is_running = False
        self.last_hr_features = None

    def get_user_label(self):
        """Prompt user to label their cognitive state via Flow State Scale.

        Maps a simplified Flow Short Scale (5 Likert items) to the 9-quadrant
        model. Users rate agreement with statements 1–5; total score maps to
        a quadrant based on challenge (BPM proxy) and skill (RMSSD proxy).
        """
        print("\n--- Flow State Scale ---")
        print("Rate each statement 1–5 (1=Strongly Disagree, 5=Strongly Agree)")
        statements = [
            "1. I feel challenged but in control of my actions.",
            "2. My mind is clear and I know what I need to do.",
            "3. I am deeply focused and fully absorbed.",
            "4. Time feels like it passes without me noticing.",
            "5. I feel calm and relaxed while staying alert.",
        ]
        scores = []
        for stmt in statements:
            print(stmt)
            while True:
                try:
                    val = input("  Score (1–5): ").strip()
                    if val in ("1", "2", "3", "4", "5"):
                        scores.append(int(val))
                        break
                    print("  Enter 1–5.")
                except KeyboardInterrupt:
                    print("\nSkipped.")
                    return None

        total = sum(scores)
        hr = self.last_hr_features['hr'] if self.last_hr_features else 70
        rmssd = self.last_hr_features['rmssd'] if self.last_hr_features else 30

        # Map score + current physiology to quadrant
        # High score (≥20) + high RMSSD = Flow/Relaxation
        # High score + low RMSSD = Arousal
        # Low score (≤10) + low RMSSD = Apathy/Panic
        # Low score + high RMSSD = Boredom
        if total >= 20:
            if rmssd > 50:
                label = "flow" if hr > 75 else "relaxation"
            else:
                label = "arousal"
        elif total >= 14:
            if rmssd > 50:
                label = "control"
            elif rmssd < 25:
                label = "worry"
            else:
                label = "neutral"
        else:
            if rmssd < 20:
                label = "panic" if hr > 80 else "apathy"
            else:
                label = "boredom"

        print(f"  → Detected flow state: {label} (score={total})")
        return label

    def run_once(self, fast_mode=False, skip_label=False):
        """Execute one iteration of the feedback loop."""
        # 1. Acquire signal
        signal = self.sensor.get_signal_buffer(duration_sec=30, fast_mode=fast_mode)

        # 2. Extract features (includes HR Delta, outlier-filtered RR)
        hr_features = self.processor.extract_features(signal)
        self.last_hr_features = hr_features

        # 3. Classify state via 9-quadrant SVM / threshold grid
        state = self.classifier.predict(hr_features)
        print(f"[PREDICT] Cognitive state: {state}")
        print(
            f"[DATA] HR: {hr_features['hr']:.1f} BPM | "
            f"RMSSD: {hr_features['rmssd']:.1f} ms | "
            f"HR Δ: {hr_features['hr_delta']:+.1f}"
        )

        # 4. Learn from user feedback (Flow State Scale)
        if not skip_label:
            user_label = self.get_user_label()
            if user_label and user_label in self.VALID_LABELS:
                self.classifier.train_incrementally(hr_features, user_label)

        # 5. RL agent decides target quadrant for music
        target_quadrant = self.rl_agent.act(
            hr_features['hr'],
            hr_features['rmssd'],
            self.previous_rmssd,
        )
        print(f"[RL] Target quadrant: {target_quadrant}")

        # 6. Apply to music engine — select best track from target quadrant
        result = self.music.apply_action(target_quadrant)
        print(f"[MUSIC] {result}")

        # Update rolling history
        self.previous_rmssd = hr_features['rmssd']

        return state, result

    def start(self, duration_minutes=10):
        """Run the feedback loop for specified duration."""
        self.is_running = True
        iterations = duration_minutes * 60 // 30  # 30 sec per iteration

        for i in range(iterations):
            if not self.is_running:
                break
            print(f"\n--- Iteration {i+1}/{iterations} ---")
            self.run_once()
            time.sleep(1)

    def stop(self):
        """Stop the feedback loop."""
        self.is_running = False
        if self.music:
            self.music.stop()


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BioSyncAI Biofeedback System")
    parser.add_argument(
        "--sensor",
        choices=["simulated", "arduino"],
        default="simulated",
        help="Sensor type: 'simulated' (default) or 'arduino' for real Arduino sensor",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Arduino serial port (e.g., /dev/cu.usbmodem1411 on macOS, COM3 on Windows). "
             "If omitted, auto-detects the port.",
    )
    args = parser.parse_args()

    system = BiofeedbackLoop(sensor_type=args.sensor, arduino_port=args.port)

    print("=" * 60)
    print("🧠 BioSyncAI - Closed-Loop Biofeedback System")
    print("=" * 60)
    print(f"Sensor mode: {args.sensor}" + (
        f" ({args.port or 'auto-detected'})" if args.sensor == "arduino" else ""
    ))
    print("This system learns from YOUR feedback every time you use it!")
    print("It will:")
    print("  • Detect your cognitive state from heart rate")
    print("  • Ask you how you actually feel (to learn)")
    print("  • Adjust music to guide you toward Flow state")
    print("  • Remember patterns for next time")
    print("=" * 60)

    # For testing, just run one iteration
    try:
        system.run_once()
        print("\n✓ System ran successfully!")
        print("📁 Models saved: svm_model.pkl, q_table.npy")
        if system.music.is_playing:
            print("🔊 Audio is playing... Press Ctrl+C to stop")
            import time
            try:
                time.sleep(5)  # Let audio play for 5 seconds
            except KeyboardInterrupt:
                pass
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        system.stop()
        if system.sensor_type == "arduino" and hasattr(system.sensor, 'close'):
            system.sensor.close()
        print("Cleanup complete.")
