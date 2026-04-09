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
    """SVM-based cognitive state classifier with incremental learning."""
    
    def __init__(self, model_path='svm_model.pkl'):
        from sklearn.svm import SVC
        self.svm = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        self.trained = False
        self.model_path = model_path
        self.training_data = []  # Store all training examples
        self.training_labels = []
        
        if os.path.exists(model_path):
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
            # Baseline detection if not trained
            rmssd = hr_features['rmssd']
            if rmssd < 20:
                return 'stress'
            elif rmssd > 50:
                return 'flow'
            return 'neutral'
        
        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)[0]
    
    def train_incrementally(self, hr_features, label):
        """Learn from new labeled data and retrain."""
        X = self.prepare_features(hr_features)
        self.training_data.append(X[0])
        self.training_labels.append(label)
        
        if len(self.training_data) >= 3:  # Need at least 3 samples
            X_train = np.array(self.training_data)
            y_train = np.array(self.training_labels)
            
            # Fit scaler and transform
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Retrain SVM
            self.svm.fit(X_train_scaled, y_train)
            self.trained = True
            self.save_model()
            print(f"✓ Model retrained with {len(self.training_data)} samples")
    
    def save_model(self):
        """Save trained model and scaler to disk."""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'svm': self.svm,
                'scaler': self.scaler,
                'training_data': self.training_data,
                'training_labels': self.training_labels,
                'trained': self.trained
            }, f)
    
    def load_model(self, model_path=None):
        """Load previously trained model."""
        path = model_path or self.model_path
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.svm = data['svm']
                self.scaler = data['scaler']
                self.training_data = data['training_data']
                self.training_labels = data['training_labels']
                self.trained = data['trained']
                print(f"✓ Loaded saved model with {len(self.training_data)} training samples")


class RLMusicAgent:
    """Reinforcement Learning agent for music parameter control with persistence."""
    
    def __init__(self, state_space_size=4, action_space_size=4, q_table_path='q_table.npy'):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.q_table_path = q_table_path
        
        # Load existing Q-table or create new
        if os.path.exists(q_table_path):
            self.q_table = np.load(q_table_path)
            print(f"✓ Loaded Q-table from previous sessions")
        else:
            self.q_table = np.zeros((state_space_size, action_space_size))
        
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        
        # Action space: which music parameter to adjust
        self.actions = ['tempo_up', 'tempo_down', 'brightness_up', 'brightness_down']
        
        # Target HRV for "Flow" state
        self.target_rmssd = 50.0
    
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
        
        # Save after each update
        self.save_q_table()
    
    def save_q_table(self):
        """Save Q-table to disk."""
        np.save(self.q_table_path, self.q_table)
    
    def act(self, current_rmssd, previous_rmssd):
        """Main RL decision loop."""
        state = self.state_from_hrv(current_rmssd)
        action_idx = self.select_action(state)
        
        reward = self.get_reward(current_rmssd, previous_rmssd)
        
        next_state = self.state_from_hrv(previous_rmssd)
        self.update_q_table(state, action_idx, reward, next_state)
        
        return self.actions[action_idx]


# =============================================================================
# 4. GENERATIVE MUSIC ENGINE (Pure Python audio synthesis)
# =============================================================================

class MusicEngine:
    """Generates/modulates music in real-time using pure Python synthesis."""
    
    def __init__(self, sample_rate=44100):
        from scipy.io import wavfile
        self.wavfile = wavfile
        self.sample_rate = sample_rate
        self.is_playing = False
        self.audio_dir = "biofeedback_audio"
        
        # Create audio directory if it doesn't exist
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
        
        # Music parameters
        self.tempo = 90
        self.brightness = 0.5
        self.rhythmic_density = 0.5
        self.harmonic_complexity = 0.5
    
    def generate_tone(self, frequency, duration_sec, amplitude=0.3):
        """Generate a sine wave tone."""
        num_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples)
        
        # Base sine wave
        wave = np.sin(2.0 * np.pi * frequency * t) * amplitude
        
        # Add harmonics based on complexity
        if self.harmonic_complexity > 0.3:
            wave += 0.5 * np.sin(2.0 * np.pi * frequency * 1.5 * t) * amplitude * self.harmonic_complexity
        if self.harmonic_complexity > 0.6:
            wave += 0.3 * np.sin(2.0 * np.pi * frequency * 2 * t) * amplitude * self.harmonic_complexity
        
        # Apply brightness (frequency enhancement)
        if self.brightness > 0.5:
            wave += 0.2 * np.sin(2.0 * np.pi * frequency * 2.5 * t * self.brightness) * amplitude
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val * 0.9
        
        # Convert to int16 for WAV format
        audio_data = np.int16(wave * 32767)
        return audio_data
    
    def generate_adaptive_music(self, duration_sec=10):
        """Generate longer, evolving adaptive music (5-10 seconds)."""
        num_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples)
        
        # Base frequency varies with brightness (dynamic pitch range)
        base_freq = 100 + (self.brightness * 120)  # 100-220 Hz range
        
        # Add subtle frequency modulation (vibrato effect)
        vibrato_rate = 5 + (self.harmonic_complexity * 10)  # 5-15 Hz
        freq_variation = base_freq * (0.98 + 0.02 * np.sin(2 * np.pi * vibrato_rate * t))
        
        # Main melody with evolving frequency
        wave = np.sin(2.0 * np.pi * freq_variation * t) * 0.3
        
        # Add chord structure (3 voices with harmonic relationship)
        if self.harmonic_complexity > 0.4:
            # 5th harmonic
            wave += 0.2 * np.sin(2.0 * np.pi * (base_freq * 1.5) * t) * self.harmonic_complexity
            # Octave
            wave += 0.15 * np.sin(2.0 * np.pi * (base_freq * 2) * t) * self.harmonic_complexity
        
        # Brightness affects high-frequency content
        if self.brightness > 0.5:
            high_freq_component = (self.brightness - 0.5) * 2  # 0-1 range
            wave += 0.1 * high_freq_component * np.sin(2.0 * np.pi * (base_freq * 3.5) * t)
        
        # Add gentle amplitude envelope (fade in/out for smooth listening)
        envelope = np.ones_like(t)
        fade_samples = int(self.sample_rate * 0.5)  # 0.5 sec fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        wave *= envelope
        
        # Normalize
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val * 0.85
        
        # Convert to int16
        audio_data = np.int16(wave * 32767)
        return audio_data
    
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
        
        # Left ear: base frequency
        left_freq = 200
        # Right ear: base + target beat frequency (creates binaural effect)
        right_freq = left_freq + target_hz
        
        # Generate stereo binaural beats
        left_channel = np.sin(2.0 * np.pi * left_freq * t) * 0.2
        right_channel = np.sin(2.0 * np.pi * right_freq * t) * 0.2
        
        # Add ambient pad (smooth background harmonics)
        ambient_freq = 110  # Base note
        ambient = 0.15 * (
            np.sin(2.0 * np.pi * ambient_freq * t) +
            0.5 * np.sin(2.0 * np.pi * ambient_freq * 1.5 * t) +
            0.3 * np.sin(2.0 * np.pi * ambient_freq * 2 * t)
        )
        
        # Combine: binaural beats + ambient pad
        left_channel += ambient
        right_channel += ambient
        
        # Smooth fade in/out
        envelope = np.ones_like(t)
        fade_samples = int(self.sample_rate * 1.0)  # 1 sec fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        left_channel *= envelope
        right_channel *= envelope
        
        # Normalize both channels
        max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        if max_val > 0:
            left_channel = left_channel / max_val * 0.8
            right_channel = right_channel / max_val * 0.8
        
        # Combine into stereo
        stereo_audio = np.column_stack((left_channel, right_channel))
        audio_data = np.int16(stereo_audio * 32767)
        
        return audio_data
    
    def save_tone(self, filename_prefix="tone"):
        """Generate and save binaural beats + ambient music."""
        try:
            # Determine target frequency based on HRV state
            rmssd_estimate = 50  # Default (would come from actual HRV in real use)
            
            if self.brightness < 0.3:
                # Stressed: use 40 Hz (focus/alerting)
                target_hz = 40
                state = "Alert/Focus"
            elif self.brightness < 0.6:
                # Moderate: use 14 Hz (focused calm)
                target_hz = 14
                state = "Focused Calm"
            else:
                # Relaxed: use 10 Hz (deep relaxation)
                target_hz = 10
                state = "Deep Relaxation"
            
            print(f"  🎵 Generating binaural beats ({state}, {target_hz}Hz)...")
            
            # Generate binaural + ambient music
            audio_data = self.generate_binaural_beats(duration_sec=10, target_hz=target_hz)
            
            # Save to file (stereo)
            filepath = os.path.join(self.audio_dir, f"{filename_prefix}.wav")
            self.wavfile.write(filepath, self.sample_rate, audio_data)
            
            print(f"  ✨ Binaural profile: {state} | Brightness: {self.brightness:.2f}")
            print(f"  🔊 Saved: {filepath}")
            
            # Auto-play
            try:
                abs_path = os.path.abspath(filepath)
                if os.name == 'nt':  # Windows
                    import subprocess
                    subprocess.Popen(['start', abs_path], shell=True)
                    print(f"  ▶️  Playing binaural beats + ambient (10 seconds)...")
            except Exception as e:
                pass
            
            return filepath
        except Exception as e:
            print(f"  ⚠️  Could not generate audio: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def adjust_tempo(self, delta):
        """Adjust tempo in BPM."""
        old_tempo = self.tempo
        self.tempo = max(60, min(180, self.tempo + delta))
        if self.tempo != old_tempo:
            print(f"  🎵 Tempo: {old_tempo} → {self.tempo} BPM")
    
    def adjust_brightness(self, delta):
        """Adjust brightness (frequency shift)."""
        old_brightness = self.brightness
        self.brightness = max(0.0, min(1.0, self.brightness + delta))
        if self.brightness != old_brightness:
            print(f"  ✨ Brightness: {old_brightness:.2f} → {self.brightness:.2f}")
    
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
        
        # Generate audio file reflecting new parameters
        self.save_tone(f"adaptive_{int(time.time())}")
        
        return self.get_current_params()
    
    def get_current_params(self):
        """Return current music parameters."""
        return {
            'tempo': self.tempo,
            'brightness': self.brightness,
            'rhythmic_density': self.rhythmic_density,
            'harmonic_complexity': self.harmonic_complexity
        }
    
    def stop(self):
        """Cleanup."""
        pass


# =============================================================================
# 5. MAIN CLOSED-LOOP SYSTEM
# =============================================================================

class BiofeedbackLoop:
    """Main closed-loop biofeedback system with learning."""
    
    def __init__(self):
        self.sensor = PulseSensorReader()
        self.processor = SignalProcessor()
        self.classifier = CognitiveStateClassifier()
        self.rl_agent = RLMusicAgent()
        self.music = MusicEngine()
        
        self.previous_rmssd = 50.0
        self.is_running = False
        self.last_hr_features = None
    
    def get_user_label(self):
        """Prompt user to label their current cognitive state."""
        print("\nHow do you feel right now?")
        print("1. Stressed")
        print("2. Neutral")
        print("3. Focused/Flow")
        print("4. Relaxed")
        
        while True:
            try:
                choice = input("Enter 1-4: ").strip()
                mapping = {'1': 'stress', '2': 'neutral', '3': 'flow', '4': 'relaxed'}
                if choice in mapping:
                    return mapping[choice]
                print("Invalid choice. Try again.")
            except KeyboardInterrupt:
                print("\nSkipped labeling.")
                return None
    
    def run_once(self, fast_mode=False):
        """Execute one iteration of the feedback loop."""
        # 1. Acquire signal
        signal = self.sensor.get_signal_buffer(duration_sec=30, fast_mode=fast_mode)
        
        # 2. Extract features
        hr_features = self.processor.extract_features(signal)
        self.last_hr_features = hr_features
        
        # 3. Classify state
        state = self.classifier.predict(hr_features)
        print(f"[PREDICT] Detected state: {state}")
        print(f"[DATA] HR: {hr_features['hr']:.1f} BPM | RMSSD: {hr_features['rmssd']:.1f}")
        
        # 4. Learn from user feedback
        user_label = self.get_user_label()
        if user_label:
            self.classifier.train_incrementally(hr_features, user_label)
        
        # 5. RL decides action
        action = self.rl_agent.act(hr_features['rmssd'], self.previous_rmssd)
        print(f"[RL] Action: {action}")
        
        # 6. Apply to music engine
        music_params = self.music.apply_action(action)
        print(f"[MUSIC] Tempo: {music_params['tempo']} | Brightness: {music_params['brightness']:.2f}")
        
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
        if self.music:
            self.music.stop()


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    system = BiofeedbackLoop()
    
    print("=" * 60)
    print("🧠 BioSyncAI - Closed-Loop Biofeedback System")
    print("=" * 60)
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
        print("Cleanup complete.")
