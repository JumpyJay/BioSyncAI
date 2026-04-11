"""Test biofeedback system — skip user label for automated testing"""

import sys
import os

# Run with --skip-label so no user input is required
from biofeedback_system import BiofeedbackLoop

print("=" * 60)
print("Testing BioSyncAI with 9-quadrant SVM + RL playlist (fast mode)")
print("=" * 60)

system = BiofeedbackLoop()

try:
    print("\n📍 Running biofeedback loop iteration (fast mode, skip labels)...")
    system.run_once(fast_mode=True, skip_label=True)
    print("\n✓ Test complete!")
    if os.path.exists(system.music.audio_dir):
        files = os.listdir(system.music.audio_dir)
        print(f"📁 Audio files: {files}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    system.stop()
    print("Cleanup complete.\n")
