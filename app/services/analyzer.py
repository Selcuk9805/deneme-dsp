import librosa
import numpy as np
import pyloudnorm as pyln
import scipy.signal

BEATS_PER_BAR = 4
KICK_BAND_HZ = 160.0


def estimate_downbeat_phase(y: np.ndarray, sr: int, beat_times: np.ndarray) -> tuple[int, float]:
    """Which of the 4 beat phases (0..3) actually carries the kick/onset accent — i.e. bar 1.

    Candidate generation previously assumed the excerpt's own first loaded beat was always phase
    0 (`idx % 4 == 0`), which has no relationship to the track's actual musical bar structure —
    it's an artifact of wherever the 60s load window happened to start. Comparing average onset
    accent strength across the 4 possible phases finds the one that actually carries the kick.
    """
    if len(beat_times) < BEATS_PER_BAR * 2:
        return 0, 0.0

    sos = scipy.signal.butter(4, KICK_BAND_HZ / (sr / 2.0), btype="lowpass", output="sos")
    y_low = scipy.signal.sosfilt(sos, y)

    onset_low = librosa.onset.onset_strength(y=y_low, sr=sr, aggregate=np.mean)
    onset_full = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)

    beat_frames_onset = np.clip(librosa.time_to_frames(beat_times, sr=sr), 0, len(onset_low) - 1)
    low_at_beat = onset_low[beat_frames_onset]
    full_at_beat = onset_full[np.clip(beat_frames_onset, 0, len(onset_full) - 1)]

    low_norm = low_at_beat / (low_at_beat.max() + 1e-9)
    full_norm = full_at_beat / (full_at_beat.max() + 1e-9)
    accents = 0.7 * low_norm + 0.3 * full_norm

    phase_scores = [accents[phase::BEATS_PER_BAR].mean() for phase in range(BEATS_PER_BAR)]
    best_phase = int(np.argmax(phase_scores))
    other_mean = float(np.mean([s for i, s in enumerate(phase_scores) if i != best_phase]) + 1e-9)
    confidence = float(np.clip(1 - other_mean / (phase_scores[best_phase] + 1e-9), 0.0, 1.0))
    return best_phase, confidence

def get_camelot_key(chroma_vals: np.ndarray) -> tuple[str, float]:
    maj_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    min_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    maj_profile = maj_profile / np.linalg.norm(maj_profile)
    min_profile = min_profile / np.linalg.norm(min_profile)
    
    if np.linalg.norm(chroma_vals) > 0:
        chroma_vals = chroma_vals / np.linalg.norm(chroma_vals)
    
    best_corr = -1
    best_key = "1A"
    
    camelot_maj = ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"]
    camelot_min = ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]
    
    for i in range(12):
        shifted_maj = np.roll(maj_profile, i)
        shifted_min = np.roll(min_profile, i)
        
        corr_maj = np.corrcoef(chroma_vals, shifted_maj)[0,1]
        corr_min = np.corrcoef(chroma_vals, shifted_min)[0,1]
        
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = camelot_maj[i]
            
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = camelot_min[i]

    # best_corr was previously computed and discarded — every key guess was treated as equally
    # trustworthy regardless of how weak/ambiguous the underlying chroma correlation actually
    # was (a near-silent or noisy excerpt still confidently "detects" a key, defaulting to "1A"
    # when nothing correlates at all). Returning it lets callers down-weight low-confidence
    # matches instead of trusting every detection equally.
    return best_key, float(best_corr)

class AudioFeatures:
    def __init__(self, y: np.ndarray, sr: int):
        self.sr = sr
        self.duration = librosa.get_duration(y=y, sr=sr)
        
        tempo_val, self.beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        self.beat_times = librosa.frames_to_time(self.beat_frames, sr=sr)

        # librosa's own `tempo` return value comes from its tempogram autocorrelation, which at
        # the default hop_length is quantized too coarsely (many distinct true BPMs collapse to
        # the same reported value). Deriving BPM from the mean spacing of the already-computed
        # beat_times sidesteps that quantization at no extra cost.
        if len(self.beat_times) >= 2:
            self.tempo = 60.0 / float(np.mean(np.diff(self.beat_times)))
        else:
            self.tempo = float(tempo_val[0]) if isinstance(tempo_val, np.ndarray) else float(tempo_val)

        self.downbeat_phase, self.downbeat_confidence = estimate_downbeat_phase(y, sr, self.beat_times)

        self.rms = librosa.feature.rms(y=y)[0]
        
        # LUFS Loudness
        meter = pyln.Meter(sr)
        try:
            self.lufs = meter.integrated_loudness(y)
        except Exception:
            self.lufs = -14.0 # Fallback
        
        # Key / Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        self.camelot_key, self.key_confidence = get_camelot_key(chroma_mean)
        
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        bass_bins = np.where((freqs >= 20) & (freqs <= 250))[0]
        total_energy = np.mean(S, axis=0)
        if len(bass_bins) > 0:
            self.bass_energy = np.mean(S[bass_bins, :], axis=0)
        else:
            self.bass_energy = np.zeros(S.shape[1])

        # Bass energy as a fraction of that frame's total spectral energy, not a raw STFT
        # magnitude. Raw magnitude scales with the track's mastering loudness, so a fixed
        # threshold on it either fires for nearly every track or almost none depending on the
        # corpus (measured: >90% of real pairs landed on the same tier — see
        # docs/decisions.md-equivalent finding in project memory). The ratio is scale-invariant.
        self.bass_energy_ratio = np.divide(
            self.bass_energy, total_energy,
            out=np.zeros_like(self.bass_energy), where=total_energy > 1e-8,
        )

        # Compute RMS and Bass energy per beat
        self.beat_energies = np.zeros(len(self.beat_frames))
        self.beat_bass_energies = np.zeros(len(self.beat_frames))

        rms_frames = librosa.feature.rms(y=y)[0]

        for i, b_frame in enumerate(self.beat_frames):
            # Window of roughly 1 beat around the frame
            start = max(0, b_frame - 10)
            end = min(len(rms_frames), b_frame + 10)
            self.beat_energies[i] = np.mean(rms_frames[start:end])

            # Bass energy window (normalized ratio, not raw magnitude)
            if len(bass_bins) > 0:
                self.beat_bass_energies[i] = np.mean(self.bass_energy_ratio[start:end])
        
        self.global_offset = 0.0

class AnalyzerService:
    @staticmethod
    def analyze_track(file_path: str, is_track_a: bool) -> AudioFeatures:
        duration = librosa.get_duration(path=file_path)
        load_duration = min(60.0, duration)
        
        if is_track_a:
            offset = max(0.0, duration - load_duration)
        else:
            offset = 0.0
            
        y, sr = librosa.load(file_path, sr=22050, offset=offset, duration=load_duration)
        features = AudioFeatures(y, sr)
        features.global_offset = offset
        
        return features
