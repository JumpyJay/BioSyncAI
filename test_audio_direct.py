"""Direct music engine test - no user input"""

import numpy as np
import os
from scipy.io import wavfile

# Setup
sample_rate = 44100
audio_dir = "biofeedback_audio_test"

if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

print("Testing audio generation directly...")

def generate_tone(frequency, duration_sec, amplitude=0.3, brightness=0.5, complexity=0.5):
    """Generate a sine wave tone."""
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples)
    
    # Base sine wave
    wave = np.sin(2.0 * np.pi * frequency * t) * amplitude
    
    # Add harmonics
    if complexity > 0.3:
        wave += 0.5 * np.sin(2.0 * np.pi * frequency * 1.5 * t) * amplitude * complexity
    if complexity > 0.6:
        wave += 0.3 * np.sin(2.0 * np.pi * frequency * 2 * t) * amplitude * complexity
    
    # Apply brightness
    if brightness > 0.5:
        wave += 0.2 * np.sin(2.0 * np.pi * frequency * 2.5 * t * brightness) * amplitude
    
    # Normalize
    max_val = np.max(np.abs(wave))
    if max_val > 0:
        wave = wave / max_val * 0.9
    
    # Convert to int16
    audio_data = np.int16(wave * 32767)
    return audio_data

# Test 1: Low brightness tone (dark/bass-heavy)
print("\n1. Generating LOW brightness tone (110 Hz, dark)...")
audio1 = generate_tone(110, 2.0, brightness=0.2, complexity=0.3)
path1 = os.path.join(audio_dir, "tone_dark.wav")
wavfile.write(path1, sample_rate, audio1)
print(f"   ✓ Saved: {path1}")
print(f"   ✓ Size: {os.path.getsize(path1)} bytes")

# Test 2: High brightness tone (bright/treble-rich)
print("\n2. Generating HIGH brightness tone (220 Hz, bright)...")
audio2 = generate_tone(220, 2.0, brightness=0.9, complexity=0.8)
path2 = os.path.join(audio_dir, "tone_bright.wav")
wavfile.write(path2, sample_rate, audio2)
print(f"   ✓ Saved: {path2}")
print(f"   ✓ Size: {os.path.getsize(path2)} bytes")

# Test 3: Smooth transition
print("\n3. Generating smooth transition (fade up in frequency)...")
duration = 3.0
num_samples = int(sample_rate * duration)
t = np.linspace(0, duration, num_samples)
freq_sweep = 110 + (220 - 110) * (t / duration)  # Sweep from 110 to 220 Hz

wave = np.sin(2.0 * np.pi * freq_sweep * t) * 0.3
max_val = np.max(np.abs(wave))
if max_val > 0:
    wave = wave / max_val * 0.9
audio3 = np.int16(wave * 32767)

path3 = os.path.join(audio_dir, "tone_sweep.wav")
wavfile.write(path3, sample_rate, audio3)
print(f"   ✓ Saved: {path3}")
print(f"   ✓ Size: {os.path.getsize(path3)} bytes")

print("\n✓ All audio files generated successfully!")
print(f"\nTo play the audio files, use:")
print(f"  - Windows: start {audio_dir}\\tone_dark.wav")
print(f"  - Or open them in any media player")
print(f"\nFiles created in: {os.path.abspath(audio_dir)}")
