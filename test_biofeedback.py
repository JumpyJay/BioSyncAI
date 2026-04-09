"""Test biofeedback system with simulated user input"""

import sys
import os

# Create a mock input function
user_inputs = iter(['1', '1', '1'])  # Always respond: "1" (Stressed)

def mock_input(prompt):
    try:
        response = next(user_inputs)
        print(prompt, response)
        return response
    except StopIteration:
        raise KeyboardInterrupt("Out of test inputs")

# Replace input with mock
import builtins
builtins.input = mock_input

# Now run the system
from biofeedback_system import BiofeedbackLoop

print("=" * 60)
print("🧪 Testing BioSyncAI with simulated user input...")
print("=" * 60)

system = BiofeedbackLoop()

try:
    print("\n📍 Running biofeedback loop iteration (fast mode)...")
    system.run_once(fast_mode=True)
    print("\n✓ Test complete!")
    print(f"📁 Audio files: {os.listdir(system.music.audio_dir)}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    system.stop()
    print("Cleanup complete.\n")
