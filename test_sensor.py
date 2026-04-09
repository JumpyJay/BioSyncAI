"""
Pulse Sensor Connection Test
Tests if your pulse sensor hardware is properly connected and communicating.
"""

import sys
import time

print("=" * 60)
print("🩺 Pulse Sensor Connection Test")
print("=" * 60)

# Test 1: Check for required libraries
print("\n[1] Checking dependencies...")
try:
    import numpy as np
    print("  ✓ numpy")
except ImportError as e:
    print(f"  ✗ numpy: {e}")

try:
    import scipy
    print("  ✓ scipy")
except ImportError as e:
    print(f"  ✗ scipy: {e}")

try:
    import sklearn
    print("  ✓ scikit-learn")
except ImportError as e:
    print(f"  ✗ scikit-learn: {e}")

# Test 2: Check for hardware communication libraries
print("\n[2] Checking hardware libraries...")

# For Raspberry Pi with ADC
try:
    import spidev
    print("  ✓ spidev (SPI communication available)")
    has_spidev = True
except ImportError:
    print("  ⚠️  spidev not installed (needed for Raspberry Pi ADC)")
    has_spidev = False

# For direct GPIO
try:
    import RPi.GPIO as GPIO
    print("  ✓ RPi.GPIO (Raspberry Pi GPIO available)")
    has_gpio = True
except ImportError:
    print("  ⚠️  RPi.GPIO not installed (Raspberry Pi only)")
    has_gpio = False

# For USB serial sensors
try:
    import serial
    print("  ✓ pyserial (USB/Serial communication available)")
    has_serial = True
except ImportError:
    print("  ⚠️  pyserial not installed (needed for USB sensors)")
    has_serial = False

# Test 3: Check for actual sensor hardware
print("\n[3] Detecting connected hardware...")

if has_spidev:
    print("  🔍 Checking SPI devices (Raspberry Pi ADC)...")
    try:
        import spidev
        spi = spidev.SpiDev()
        spi.open(0, 0)  # Bus 0, Device 0 (MCP3008)
        print(f"    ✓ SPI device found: /dev/spidev0.0")
        print(f"    ✓ Attempting to read from channel 0...")
        
        # Try reading a value
        response = spi.xfer2([1, 0xA0, 0])  # MCP3008 read command
        value = ((response[1] & 3) << 8) + response[2]
        print(f"    ✓ Channel 0 reading: {value} (0-1023 scale)")
        print(f"    ✓ Voltage: {(value / 1023.0 * 3.3):.2f}V")
        spi.close()
        
    except Exception as e:
        print(f"    ✗ SPI read failed: {e}")
        print(f"    💡 Make sure MCP3008 ADC is properly connected on SPI0")

if has_gpio:
    print("  🔍 Checking GPIO pins (Raspberry Pi)...")
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        print(f"    ✓ GPIO interface ready")
        print(f"    💡 Pulse sensor typically uses GPIO pin 17 or 27")
        GPIO.cleanup()
    except Exception as e:
        print(f"    ✗ GPIO access failed: {e}")
        print(f"    💡 Run with sudo: sudo python test_sensor.py")

if has_serial:
    print("  🔍 Checking USB/Serial devices...")
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    if ports:
        print(f"    ✓ Found {len(ports)} serial ports:")
        for port in ports:
            print(f"      - {port.device}: {port.description}")
            try:
                ser = serial.Serial(port.device, 9600, timeout=1)
                print(f"        ✓ Connected at 9600 baud")
                data = ser.readline()
                if data:
                    print(f"        ✓ Data received: {data[:50]}")
                ser.close()
            except Exception as e:
                print(f"        ⚠️  Could not read from {port.device}: {e}")
    else:
        print(f"    ⚠️  No serial ports detected")

# Test 4: Current sensor simulation (fallback)
print("\n[4] Simulated sensor test...")
from biofeedback_system import PulseSensorReader

sensor = PulseSensorReader(adc_channel=0, sample_rate=100)
print(f"  🔧 Created PulseSensorReader (sample_rate=100Hz)")

# Take a quick sample
print(f"  📊 Collecting 5 samples...")
for i in range(5):
    raw = sensor.read_raw()
    print(f"    Sample {i+1}: {raw:.2f}V")
    time.sleep(0.1)

# Test 5: Summary
print("\n" + "=" * 60)
print("📋 SENSOR CONNECTION SUMMARY")
print("=" * 60)

if has_spidev or has_gpio or has_serial:
    print("✅ Hardware libraries available")
    print("   Your system can communicate with external sensors")
else:
    print("⚠️  No hardware communication libraries detected")
    print("   Currently using simulated sensor data")
    print("\n💡 To use real hardware, install:")
    if not has_spidev:
        print("   • Raspberry Pi ADC: pip install spidev")
    if not has_gpio:
        print("   • Raspberry Pi GPIO: pip install RPi.GPIO")
    if not has_serial:
        print("   • USB sensors: pip install pyserial")

print("\n🩺 Next steps:")
print("   1. Verify pulse sensor is plugged in")
print("   2. Check ADC/GPIO connections")
print("   3. Run: python biofeedback_system.py")
print("=" * 60)
