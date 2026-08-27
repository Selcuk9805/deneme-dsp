import numpy as np
import soundfile as sf

sr = 22050
duration = 65.0
t = np.linspace(0, duration, int(sr * duration), False)

# Track A: 120 BPM kick pattern + sine wave
# 120 BPM = 2 beats per second
kick_freq = 2.0
kick_env = np.maximum(0, np.sin(2 * np.pi * kick_freq * t))**20
audio_a = 0.5 * np.sin(2 * np.pi * 440 * t) + kick_env * 0.8 * np.sin(2 * np.pi * 60 * t)
sf.write("trackA.wav", audio_a, sr)

# Track B: 122 BPM kick pattern + noise
kick_freq_b = 122.0 / 60.0
kick_env_b = np.maximum(0, np.sin(2 * np.pi * kick_freq_b * t))**20
audio_b = 0.3 * np.random.randn(len(t)) + kick_env_b * 0.8 * np.sin(2 * np.pi * 80 * t)
sf.write("trackB.wav", audio_b, sr)

print("Test audio generated.")
