"""
Closed-Loop Biofeedback Music System
=====================================
A system that reads physiological signals, classifies cognitive state,
and generates adaptive music in real-time.
"""

import numpy as np
from collections import deque
import time


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
    
    def get_signal_buffer(self, duration_sec=10):
        """Collect signal for specified duration."""
        samples_needed = self.sample_rate * duration_sec
        signal = []
        while len(signal) < samples_needed:
            signal.append(self.read_raw())
            time.sleep(1 / self.sample_rate)
        return np.array(signal)


# =============================================================================
# 2. SIGNAL PROCESSING & FEATURE EXTRACTION
# =============================================================================

class SignalProcessor:
    """Extract HRV features from raw pulse signal."""
    
    def __init__(self, sample_rate=100):
        self.sample_rate = sample_rate
    
    def detect_peaks(self, signal, min_distance=300):
        """Detect R-peaks in pulse signal."""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(signal, distance=min_distance)
        return peaks
    
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
    
    def extract_features(self, signal):
        """Extract all features for SVM classification."""
        peaks = self.detect_peaks(signal)
        rr_intervals = self.calculate_rr_intervals(peaks)
        
        features = {
            'hr': self.calculate_hr(rr_intervals),
            'rmssd': self.calculate_rmssd(rr_intervals),
            'rr_mean': np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
            'rr_std': np.std(rr_intervals) if len(rr_intervals) > 0 else 0,
        }
        return features


# =============================================================================
# 3. INTELLIGENCE LAYER (SVM + RL)
# =============================================================================

class CognitiveStateClassifier:
    """SVM-based cognitive state classifier."""
    
    def __init__(self, model_path=None):
        from sklearn.svm import SVC
        self.svm = SVC(kernel='rbf', probability=True)
        self.trained = False
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_features(self, hr_features):
        """Format features for SVM input."""
        return np.array([
            hr_features['hr'],
            hr_features['rmssd'],
            hr_features['rr_std']
        ]).reshape(1, -1)
    
    def predict(self, hr_features):
        """Predict cognitive state."""
        X = self.prepare_features(hr_features)
        
        if not self.trained:
            # Placeholder: return baseline state
            # In production, load pre-trained WESAD model
            rmssd = hr_features['rmssd']
            if rmssd < 20:
                return 'stress'
            elif rmssd > 50:
                return 'flow'
            return 'neutral'
        
        return self.svm.predict(X)[0]
    
    def train(self, X_train, y_train):
        """Train SVM on labeled data (e.g., WESAD dataset)."""
        self.svm.fit(X_train, y_train)
        self.trained = True


class RLMusicAgent:
    """Reinforcement Learning agent for music parameter control."""
    
    def __init__(self, state_space_size=4, action_space_size=4):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        
        # Action space: which music parameter to adjust
        self.actions = ['tempo_up', 'tempo_down', 'brightness_up', 'brightness_down']
        
        # Target HRV for "Flow" state
        self.target_rmssd = 50.0

        # Memory for Q-learning updates
        self.last_state = None
        self.last_action_idx = None
    
    def state_from_hrv(self, rmssd):
        """Discretize HRV into state bins."""
        if rmssd < 20:
            return 0  # stress
        elif rmssd < 40:
            return 1  # moderate
        elif rmssd < 60:
            return 2  # relaxed
        return 3  # flow
    
    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        return np.argmax(self.q_table[state])
    
    def get_reward(self, current_rmssd, previous_rmssd):
        """Reward based on HRV movement toward Flow state."""
        distance_to_target = abs(self.target_rmssd - current_rmssd)
        prev_distance = abs(self.target_rmssd - previous_rmssd)
        
        if distance_to_target < prev_distance:
            return 1.0  # Moving toward flow
        elif distance_to_target > prev_distance:
            return -1.0  # Moving away from flow
        return 0.0  # No change
    
    def update_q_table(self, state, action, reward, next_state):
        """Q-learning update."""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        self.q_table[state, action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
    
    def act(self, current_rmssd, previous_rmssd):
        """Main RL decision loop with state tracking."""
        current_state = self.state_from_hrv(current_rmssd)
        
        # Update Q-table based on the transition from the previous iteration
        if self.last_state is not None and self.last_action_idx is not None:
            reward = self.get_reward(current_rmssd, previous_rmssd)
            self.update_q_table(self.last_state, self.last_action_idx, reward, current_state)
        
        # Select next action
        action_idx = self.select_action(current_state)
        
        # Store for next iteration
        self.last_state = current_state
        self.last_action_idx = action_idx
        
        return self.actions[action_idx]


# =============================================================================
# 4. GENERATIVE MUSIC ENGINE
# =============================================================================

class MusicEngine:
    """Generates/modulates music based on biofeedback."""
    
    def __init__(self):
        # Placeholder: would integrate with Tone.js, SuperCollider, or pyo
        self.tempo = 90
        self.brightness = 0.5
        self.rhythmic_density = 0.5
        self.harmonic_complexity = 0.5
    
    def adjust_tempo(self, delta):
        """Adjust tempo in BPM."""
        self.tempo = max(60, min(180, self.tempo + delta))
    
    def adjust_brightness(self, delta):
        """Adjust brightness (harmonic brightness/filter cutoff)."""
        self.brightness = max(0.0, min(1.0, self.brightness + delta))
    
    def adjust_rhythmic_density(self, delta):
        """Adjust rhythmic density (note density per measure)."""
        self.rhythmic_density = max(0.0, min(1.0, self.rhythmic_density + delta))
    
    def adjust_harmonic_complexity(self, delta):
        """Adjust harmonic complexity."""
        self.harmonic_complexity = max(0.0, min(1.0, self.harmonic_complexity + delta))
    
    def apply_action(self, action):
        """Apply RL action to music parameters."""
        action_map = {
            'tempo_up': lambda: self.adjust_tempo(5),
            'tempo_down': lambda: self.adjust_tempo(-5),
            'brightness_up': lambda: self.adjust_brightness(0.1),
            'brightness_down': lambda: self.adjust_brightness(-0.1),
        }
        
        if action in action_map:
            action_map[action]()
        
        return self.get_current_params()
    
    def get_current_params(self):
        """Return current music parameters."""
        return {
            'tempo': self.tempo,
            'brightness': self.brightness,
            'rhythmic_density': self.rhythmic_density,
            'harmonic_complexity': self.harmonic_complexity
        }


# =============================================================================
# 5. MAIN CLOSED-LOOP SYSTEM
# =============================================================================

class BiofeedbackLoop:
    """Main closed-loop biofeedback system."""
    
    def __init__(self):
        self.sensor = PulseSensorReader()
        self.processor = SignalProcessor()
        self.classifier = CognitiveStateClassifier()
        self.rl_agent = RLMusicAgent()
        self.music = MusicEngine()
        
        self.previous_rmssd = 50.0
        self.is_running = False
    
    def run_once(self):
        """Execute one iteration of the feedback loop."""
        # 1. Acquire signal
        signal = self.sensor.get_signal_buffer(duration_sec=30)
        
        # 2. Extract features
        hr_features = self.processor.extract_features(signal)
        
        # 3. Classify state
        state = self.classifier.predict(hr_features)
        print(f"Detected state: {state}")
        
        # 4. RL decides action
        action = self.rl_agent.act(hr_features['rmssd'], self.previous_rmssd)
        print(f"RL Action: {action}")
        
        # 5. Apply to music engine
        music_params = self.music.apply_action(action)
        print(f"Music params: {music_params}")
        
        # Update for next iteration
        self.previous_rmssd = hr_features['rmssd']
        
        return state, music_params
    
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


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    system = BiofeedbackLoop()
    
    # For testing, just run one iteration
    system.run_once()
