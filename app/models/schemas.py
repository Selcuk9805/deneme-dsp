from pydantic import BaseModel
from typing import List, Optional, Literal

class TransitionRequest(BaseModel):
    track_a: str
    track_b: str

class AutomationEvent(BaseModel):
    execution_time: float
    value: float
    type: Literal["set", "fadeVolume"]
    curve: Optional[Literal["linear", "equal_power", "cut"]] = "linear"

class BiquadFilterEvent(BaseModel):
    filter_type: Literal["highpass", "lowpass", "bandpass", "lowshelf", "highshelf", "peaking", "notch", "allpass"]
    parameter: Literal["frequency", "resonance"]
    execution_time: float
    value: float
    type: Literal["set", "fadeFilterParameter"]
    curve: Optional[Literal["linear", "equal_power", "cut"]] = "linear"

class TrackAutomation(BaseModel):
    lufs_gain_db: Optional[float] = 0.0
    camelot_key: Optional[str] = None
    volume: List[AutomationEvent]
    biquad_filters: Optional[List[BiquadFilterEvent]] = []

class AutomationInfo(BaseModel):
    track_a: TrackAutomation
    track_b: TrackAutomation

class TimingInfo(BaseModel):
    transition_duration_source: float
    transition_duration_execution: float
    track_a_start_crossfade_source: float
    track_b_start_source: float
    track_b_play_delay_execution: float

class BeatAlignment(BaseModel):
    track_a_beat_sample: int
    track_b_beat_sample: int
    alignment_confidence: float

class SyncInfo(BaseModel):
    target_bpm: float
    track_a_speed_ratio: float
    track_b_speed_ratio: float
    beat_alignment: BeatAlignment

class TransitionScores(BaseModel):
    tempo_compatibility: float
    key_compatibility: float
    phrase_compatibility: float
    energy_compatibility: float
    bass_compatibility: float

class TransitionCandidate(BaseModel):
    id: str
    score: float

class TransitionDecision(BaseModel):
    strategy: Literal["phrase_crossfade", "harmonic_crossfade", "standard_crossfade", "short_crossfade", "beat_cut", "hard_cut"]
    selected_candidate_id: str
    score: float
    confidence: float
    scores: TransitionScores
    candidates: Optional[List[TransitionCandidate]] = []

class TransitionPlan(BaseModel):
    status: str
    schema_version: int = 3
    decision: TransitionDecision
    sync: SyncInfo
    timing: TimingInfo
    automation: AutomationInfo
