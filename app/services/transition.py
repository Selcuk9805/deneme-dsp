from app.services.analyzer import AudioFeatures
from app.services.timeline import TimelineService
from app.models.schemas import (
    TransitionPlan, SyncInfo, TimingInfo, AutomationInfo, TrackAutomation,
    AutomationEvent, BiquadFilterEvent, TransitionDecision, TransitionScores,
    TransitionCandidate, BeatAlignment
)
import numpy as np

# beat_bass_energies is a scale-invariant ratio (bass-band STFT magnitude / total spectral
# magnitude for that frame), not raw magnitude — see analyzer.py. Thresholds calibrated from the
# combined distribution (~47k beat samples) across BOTH the progressive-house (beatport) and
# Turkish-pop (pop_corpus) corpora (tools/analyze_corpus.py) — not just one genre, since bass
# prominence differs systematically by genre (EDM medians ~15.5, pop medians ~10.4) and a
# single-genre calibration didn't transfer: tuning on beatport alone left pop_corpus with almost
# no "both loud" pairs (13/20306). ~p65/~p25 of the combined distribution. Chosen so "both loud" /
# "both quiet" each cover a meaningful, non-degenerate share of real pairs in both corpora instead
# of one tier absorbing ~94% of them (the old raw-magnitude thresholds 0.1/0.05 did, since raw
# STFT magnitude scales with a track's mastering loudness rather than actual bass prominence).
LOUD_BASS_THRESHOLD = 12.5
QUIET_BASS_THRESHOLD = 7.5

# Candidate-score margin -> confidence. A tie with the runner-up candidate (margin=0) means the
# choice was genuinely ambiguous (confidence 0.5); a decisive lead saturates toward the ceiling.
# Floor/ceiling keep this from ever claiming total certainty or total doubt about a decision that
# was still made deterministically. Scale calibrated from the real margin distribution measured
# across the beatport corpus (tools/analyze_corpus.py): median margin is ~0.001 (candidates on
# the bar-boundary grid are frequently near-identical — a genuine tie, not a measurement gap),
# p95 ~0.027. Scale=15 maps that p95 to confidence ~0.9, so confidence actually spans a
# meaningful range instead of clustering just above the floor for nearly every pair (an earlier
# scale=5 did — max observed confidence was 0.675 across a sample where the ceiling is 0.98).
CONFIDENCE_MARGIN_SCALE = 15.0
CONFIDENCE_FLOOR = 0.3
CONFIDENCE_CEIL = 0.98

# How many beats forward from a candidate point to average beat_energies over, instead of that
# single beat's instantaneous reading — a candidate can look energetic at the exact instant and
# still be one bar away from a breakdown/lull right after it. 4 beats = 1 bar, the smallest real
# crossfade this project produces (short_crossfade's fade_beats).
ENERGY_LOOKAHEAD_BEATS = 4

# get_camelot_key's best_corr (returned as AudioFeatures.key_confidence), measured across 148
# beatport analyses: min=0.21 max=0.91 median=0.52, p10=0.28. Threshold set just above p10 so
# roughly the bottom ~10% of individually weak detections get neutralized — consistent with the
# ~12% "fell back to the default '1A' key" rate measured in the pre-confidence baseline (those
# are exactly the near-zero-correlation cases this threshold is meant to catch).
KEY_CONFIDENCE_THRESHOLD = 0.3

# Strategy -> crossfade length, in BARS (4 beats each) not raw beats. The original values here
# (16/8/4/2) were used directly as a beat count, so "harmonic_crossfade" — meant to be the
# longest, most confident blend — worked out to just 4 bars (~7.7s at 124 BPM), barely longer
# than a single "standard_crossfade" bar-group and far short of a real DJ's extended blend for an
# excellent match. Rescaled to real bar counts (a genuine user report from real-device listening:
# "önceki projelerimize göre mix süresi belirgin şekilde daha kısa" — confirmed against the
# sibling project's own proven 4-8-bar, up-to-20s-for-an-excellent-match blend lengths).
# harmonic_crossfade uses 12 (not 16) bars specifically so it stays meaningfully differentiated
# from phrase_crossfade across most real tempos instead of both simply hitting
# MAX_CROSSFADE_SECONDS — at 16 bars, anything under ~160 BPM would clamp to the same 24s ceiling.
BEATS_PER_BAR = 4
FADE_BARS = {"harmonic_crossfade": 12.0, "phrase_crossfade": 8.0, "standard_crossfade": 4.0, "short_crossfade": 2.0, "beat_cut": 0.0}
MIN_CROSSFADE_SECONDS = 2.0
MAX_CROSSFADE_SECONDS = 24.0

# estimate_downbeat_phase's raw confidence sits on a much smaller natural scale than the 0.5-1.0
# range decision.confidence uses (four beat-phases sharing real music's onset energy fairly
# evenly means even a *correct* phase pick rarely dominates outright) — measured pairwise min
# (the weakest-link value alignment_confidence actually reports) across 5402 real beatport pairs:
# median ~0.09, p95 ~0.25, max ~0.57. Rescaled the same way decision.confidence was so the
# reported value spans a meaningful range instead of clustering near the floor.
ALIGNMENT_CONFIDENCE_SCALE = 2.5
ALIGNMENT_CONFIDENCE_FLOOR = 0.3
ALIGNMENT_CONFIDENCE_CEIL = 0.98


def get_key_score(key1: str, key2: str) -> float:
    if key1 == key2: return 1.0
    if key1[:-1] == key2[:-1]: return 0.9
    n1, n2 = int(key1[:-1]), int(key2[:-1])
    if key1[-1] == key2[-1] and (abs(n1 - n2) == 1 or abs(n1 - n2) == 11):
        return 0.8
    return 0.2


def _forward_energy(energies: np.ndarray, idx: int, beats: int = ENERGY_LOOKAHEAD_BEATS) -> float:
    """Mean of `energies` over the `beats` beats forward from `idx`, not just `energies[idx]`."""
    if len(energies) == 0:
        return 0.0
    start = min(idx, len(energies) - 1)
    window = energies[start:min(len(energies), start + beats)]
    return float(np.mean(window)) if len(window) > 0 else float(energies[start])


class TransitionService:
    @staticmethod
    def calculate_transition(features_a: AudioFeatures, features_b: AudioFeatures) -> TransitionPlan:
        # Sync & Tempo Score
        target_bpm, ratio_a, ratio_b = TimelineService.calculate_sync(features_a.tempo, features_b.tempo)
        tempo_diff = abs(ratio_a - 1.0) + abs(ratio_b - 1.0)
        tempo_comp = max(0.0, 1.0 - tempo_diff * 5)
        
        # Key Score — neutral (not compatible, not incompatible) when either track's key
        # detection wasn't confident enough to trust (near-silent/ambiguous chroma content),
        # instead of treating every detection as equally reliable regardless of how weak the
        # underlying chroma correlation was.
        key_reliable = (
            features_a.key_confidence >= KEY_CONFIDENCE_THRESHOLD
            and features_b.key_confidence >= KEY_CONFIDENCE_THRESHOLD
        )
        key_comp = get_key_score(features_a.camelot_key, features_b.camelot_key) if key_reliable else 0.5

        # Candidate Generation
        candidates = []
        beats_a = len(features_a.beat_times)
        beats_b = len(features_b.beat_times)

        # Generate candidates using last 32 beats of A and first 16 beats of B, aligned to each
        # track's own real downbeat phase (see analyzer.py's estimate_downbeat_phase) rather than
        # assuming the excerpt's first loaded beat is always bar 1.
        a_candidates = []
        for i in range(min(32, beats_a)):
            idx = beats_a - 1 - i
            if (idx - features_a.downbeat_phase) % 4 == 0:  # real bar boundary
                a_candidates.append(idx)

        b_candidates = []
        for i in range(min(16, beats_b)):
            if (i - features_b.downbeat_phase) % 4 == 0:
                b_candidates.append(i)
                
        if not a_candidates: a_candidates = [beats_a - 1] if beats_a > 0 else [0]
        if not b_candidates: b_candidates = [0]
        
        # Candidate Scoring
        for a_idx in a_candidates:
            for b_idx in b_candidates:
                # Phrase compatibility (prefer 8-bar / 32-beat boundaries from the end)
                dist_from_end = beats_a - a_idx
                phrase_comp = 1.0 if dist_from_end % 16 == 0 else (0.8 if dist_from_end % 8 == 0 else 0.5)
                if b_idx == 0: phrase_comp = min(1.0, phrase_comp + 0.1)
                
                # Energy compatibility — averaged forward from the candidate point, not just its
                # instantaneous reading, so a candidate about to ride into a lull scores lower.
                energy_a = _forward_energy(features_a.beat_energies, a_idx)
                energy_b = _forward_energy(features_b.beat_energies, b_idx)
                energy_comp = max(0.0, 1.0 - abs(energy_a - energy_b) * 2)
                
                # Bass compatibility
                bass_a = features_a.beat_bass_energies[a_idx] if a_idx < len(features_a.beat_bass_energies) else 0.0
                bass_b = features_b.beat_bass_energies[b_idx] if b_idx < len(features_b.beat_bass_energies) else 0.0
                bass_comp = 1.0
                if bass_a > LOUD_BASS_THRESHOLD and bass_b > LOUD_BASS_THRESHOLD:
                    bass_comp = 0.3
                elif bass_a < QUIET_BASS_THRESHOLD and bass_b < QUIET_BASS_THRESHOLD:
                    bass_comp = 0.5
                
                total = (tempo_comp * 0.15) + (key_comp * 0.25) + (phrase_comp * 0.3) + (energy_comp * 0.15) + (bass_comp * 0.15)
                
                c_id = f"A{a_idx}_B{b_idx}"
                candidates.append({
                    "id": c_id, "score": round(total, 3), "a_idx": a_idx, "b_idx": b_idx,
                    "scores": TransitionScores(
                        tempo_compatibility=round(tempo_comp, 3),
                        key_compatibility=round(key_comp, 3),
                        phrase_compatibility=round(phrase_comp, 3),
                        energy_compatibility=round(energy_comp, 3),
                        bass_compatibility=round(bass_comp, 3)
                    )
                })
                
        # Select best candidate
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        # Confidence from how decisively the best candidate beat the runner-up, not a constant.
        margin = (best["score"] - candidates[1]["score"]) if len(candidates) >= 2 else 1.0
        confidence = float(np.clip(0.5 + margin * CONFIDENCE_MARGIN_SCALE, CONFIDENCE_FLOOR, CONFIDENCE_CEIL))

        # Strategy Decision
        score = best["score"]
        if score > 0.85 and key_comp >= 0.8:
            strategy = "harmonic_crossfade"
        elif score > 0.75:
            strategy = "phrase_crossfade"
        elif score > 0.55:
            strategy = "standard_crossfade"
        elif score > 0.4:
            strategy = "short_crossfade"
        else:
            strategy = "beat_cut"
        fade_beats = FADE_BARS[strategy] * BEATS_PER_BAR
            
        a_idx = best["a_idx"]
        b_idx = best["b_idx"]
        
        # Timing calculations
        start_crossfade_relative = features_a.beat_times[a_idx] if a_idx < len(features_a.beat_times) else 0.0
        a_crossfade_source = start_crossfade_relative + features_a.global_offset
        b_start_source = features_b.beat_times[b_idx] + features_b.global_offset if b_idx < len(features_b.beat_times) else 0.0
        
        # Sample Alignment
        track_a_beat_sample = int(a_crossfade_source * features_a.sr)
        track_b_beat_sample = int(b_start_source * features_b.sr)
        
        source_duration_a = (fade_beats / features_a.tempo) * 60.0 if features_a.tempo > 0 else 0.0
        duration_execution = source_duration_a / ratio_a
        if fade_beats > 0:
            # Clamp the real-world listening duration (execution time — what the crossfade
            # actually takes to play), not the source-time value: when ratio_a < 1 (track A
            # slowed down), execution time stretches past source time, so clamping source alone
            # could still let the *actual* crossfade exceed MAX_CROSSFADE_SECONDS (measured: one
            # pop_corpus pair reached 25.7s before this fix). source_duration_a is then re-derived
            # from the clamped execution value so TimelineService's source/execution relationship
            # (execution = source / ratio_a) stays exact — player-branched inverts that formula
            # client-side (see project memory).
            duration_execution = float(np.clip(duration_execution, MIN_CROSSFADE_SECONDS, MAX_CROSSFADE_SECONDS))
            source_duration_a = duration_execution * ratio_a
        
        # Automation & DSP
        lufs_gain_a = -14.0 - features_a.lufs
        lufs_gain_b = -14.0 - features_b.lufs
        
        vol_a, vol_b, filt_a = [], [], []
        curve_type = "equal_power" if fade_beats > 0 else "cut"
        
        if fade_beats > 0:
            vol_a = [
                AutomationEvent(execution_time=0.0, value=1.0, type="set", curve="linear"),
                AutomationEvent(execution_time=duration_execution, value=0.0, type="fadeVolume", curve=curve_type)
            ]
            vol_b = [
                AutomationEvent(execution_time=0.0, value=0.0, type="set", curve="linear"),
                AutomationEvent(execution_time=duration_execution, value=1.0, type="fadeVolume", curve=curve_type)
            ]
            
            # Mild Bass Shelf/HPF if bass overlaps
            if best["scores"].bass_compatibility < 0.5:
                filt_a = [
                    BiquadFilterEvent(filter_type="highpass", parameter="frequency", execution_time=0.0, value=70.0, type="set", curve="linear"),
                    BiquadFilterEvent(filter_type="highpass", parameter="frequency", execution_time=duration_execution, value=150.0, type="fadeFilterParameter", curve="linear")
                ]
        else:
            vol_a = [
                AutomationEvent(execution_time=0.0, value=1.0, type="set", curve="linear"),
                AutomationEvent(execution_time=0.01, value=0.0, type="fadeVolume", curve="cut")
            ]
            vol_b = [
                AutomationEvent(execution_time=0.0, value=1.0, type="set", curve="linear")
            ]
            
        c_list = [TransitionCandidate(id=c["id"], score=c["score"]) for c in candidates[:5]]
        
        return TransitionPlan(
            status="success",
            schema_version=3,
            decision=TransitionDecision(
                strategy=strategy,
                selected_candidate_id=best["id"],
                score=best["score"],
                confidence=round(confidence, 3),
                scores=best["scores"],
                candidates=c_list
            ),
            sync=SyncInfo(
                target_bpm=round(target_bpm, 2),
                track_a_speed_ratio=round(ratio_a, 4),
                track_b_speed_ratio=round(ratio_b, 4),
                beat_alignment=BeatAlignment(
                    track_a_beat_sample=track_a_beat_sample,
                    track_b_beat_sample=track_b_beat_sample,
                    # weakest-link: alignment is only as trustworthy as whichever track's
                    # downbeat phase was harder to pin down. Rescaled — see ALIGNMENT_CONFIDENCE_*.
                    alignment_confidence=round(float(np.clip(
                        ALIGNMENT_CONFIDENCE_FLOOR + min(features_a.downbeat_confidence, features_b.downbeat_confidence) * ALIGNMENT_CONFIDENCE_SCALE,
                        ALIGNMENT_CONFIDENCE_FLOOR, ALIGNMENT_CONFIDENCE_CEIL,
                    )), 3)
                )
            ),
            timing=TimingInfo(
                transition_duration_source=round(source_duration_a, 4),
                transition_duration_execution=round(duration_execution, 4),
                track_a_start_crossfade_source=round(a_crossfade_source, 4),
                track_b_start_source=round(b_start_source, 4),
                track_b_play_delay_execution=0.0
            ),
            automation=AutomationInfo(
                track_a=TrackAutomation(
                    lufs_gain_db=round(lufs_gain_a, 2),
                    camelot_key=features_a.camelot_key,
                    volume=vol_a, 
                    biquad_filters=filt_a
                ),
                track_b=TrackAutomation(
                    lufs_gain_db=round(lufs_gain_b, 2),
                    camelot_key=features_b.camelot_key,
                    volume=vol_b, 
                    biquad_filters=[]
                )
            )
        )
