from app.services.analyzer import AudioFeatures
from app.services.timeline import TimelineService
from app.models.schemas import (
    TransitionPlan, SyncInfo, TimingInfo, AutomationInfo, TrackAutomation,
    AutomationEvent, BiquadFilterEvent, TransitionDecision, TransitionScores,
    TransitionCandidate, BeatAlignment
)
import numpy as np

def get_key_score(key1: str, key2: str) -> float:
    if key1 == key2: return 1.0
    if key1[:-1] == key2[:-1]: return 0.9
    n1, n2 = int(key1[:-1]), int(key2[:-1])
    if key1[-1] == key2[-1] and (abs(n1 - n2) == 1 or abs(n1 - n2) == 11):
        return 0.8
    return 0.2

class TransitionService:
    @staticmethod
    def calculate_transition(features_a: AudioFeatures, features_b: AudioFeatures) -> TransitionPlan:
        # Sync & Tempo Score
        target_bpm, ratio_a, ratio_b = TimelineService.calculate_sync(features_a.tempo, features_b.tempo)
        tempo_diff = abs(ratio_a - 1.0) + abs(ratio_b - 1.0)
        tempo_comp = max(0.0, 1.0 - tempo_diff * 5)
        
        # Key Score
        key_comp = get_key_score(features_a.camelot_key, features_b.camelot_key)
        
        # Candidate Generation
        candidates = []
        beats_a = len(features_a.beat_times)
        beats_b = len(features_b.beat_times)
        
        # Generate candidates using last 32 beats of A and first 16 beats of B
        a_candidates = []
        for i in range(min(32, beats_a)):
            idx = beats_a - 1 - i
            if idx % 4 == 0:  # Bar boundary
                a_candidates.append(idx)
                
        b_candidates = []
        for i in range(min(16, beats_b)):
            if i % 4 == 0:
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
                
                # Energy compatibility
                energy_a = features_a.beat_energies[a_idx] if a_idx < len(features_a.beat_energies) else 0.0
                energy_b = features_b.beat_energies[b_idx] if b_idx < len(features_b.beat_energies) else 0.0
                energy_comp = max(0.0, 1.0 - abs(energy_a - energy_b) * 2)
                
                # Bass compatibility
                bass_a = features_a.beat_bass_energies[a_idx] if a_idx < len(features_a.beat_bass_energies) else 0.0
                bass_b = features_b.beat_bass_energies[b_idx] if b_idx < len(features_b.beat_bass_energies) else 0.0
                bass_comp = 1.0
                if bass_a > 0.1 and bass_b > 0.1:
                    bass_comp = 0.3
                elif bass_a < 0.05 and bass_b < 0.05:
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
        
        # Strategy Decision
        score = best["score"]
        if score > 0.85 and key_comp >= 0.8:
            strategy = "harmonic_crossfade"
            fade_beats = 16.0
        elif score > 0.75:
            strategy = "phrase_crossfade"
            fade_beats = 8.0
        elif score > 0.55:
            strategy = "standard_crossfade"
            fade_beats = 4.0
        elif score > 0.4:
            strategy = "short_crossfade"
            fade_beats = 2.0
        else:
            strategy = "beat_cut"
            fade_beats = 0.0
            
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
                confidence=0.85, # static for now unless ML model adds true variance
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
                    alignment_confidence=0.95
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
