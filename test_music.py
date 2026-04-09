"""Quick test of music generation"""

import sys
sys.path.insert(0, '.')

from biofeedback_system import MusicEngine

print("Testing MusicEngine...")

# Create music engine
music = MusicEngine()

# Test 1: Generate a simple tone
print("\n1. Testing tone generation...")
try:
    audio_data = music.generate_tone(440, duration_sec=1.0)
    print(f"   ✓ Generated audio: {len(audio_data)} samples")
    print(f"   Audio dtype: {audio_data.dtype}")
    print(f"   Audio range: {audio_data.min()} to {audio_data.max()}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Save a tone
print("\n2. Testing tone save...")
try:
    result = music.save_tone("test_tone")
    if result:
        print(f"   ✓ Saved: {result}")
        import os
        if os.path.exists(result):
            size = os.path.getsize(result)
            print(f"   ✓ File size: {size} bytes")
    else:
        print(f"   ✗ save_tone returned None")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Apply action and generate music
print("\n3. Testing music action...")
try:
    music.brightness = 0.7
    music.harmonic_complexity = 0.5
    params = music.apply_action('brightness_up')
    print(f"   ✓ Applied action, params: {params}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Music engine test complete!")
print(f"\nAudio files location: {music.audio_dir}/")
import os
if os.path.exists(music.audio_dir):
    files = os.listdir(music.audio_dir)
    print(f"Generated files: {files}")
