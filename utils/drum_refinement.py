"""
Drum Refinement Module

Post-processing to expand ADTOF's 5 classes into 7+ classes:
- hihat → hihat_open / hihat_closed (based on decay analysis)
- crash → crash / ride (based on stem comparison)

Based on: https://arxiv.org/html/2509.24853v1
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


def compute_loudness_curve(
    waveform: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    frame_length: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RMS loudness curve for audio.
    
    Returns:
        times: Array of time points (seconds)
        loudness: Array of RMS values
    """
    # Ensure mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)
    
    # Compute RMS in frames
    num_frames = (len(waveform) - frame_length) // hop_length + 1
    loudness = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = waveform[start:end]
        loudness[i] = np.sqrt(np.mean(frame ** 2))
    
    # Time array
    times = np.arange(num_frames) * hop_length / sample_rate
    
    return times, loudness


def get_loudness_at_time(
    times: np.ndarray,
    loudness: np.ndarray,
    target_time: float,
) -> float:
    """Get interpolated loudness value at a specific time."""
    if target_time <= times[0]:
        return loudness[0]
    if target_time >= times[-1]:
        return loudness[-1]
    
    idx = np.searchsorted(times, target_time)
    if idx == 0:
        return loudness[0]
    
    # Linear interpolation
    t0, t1 = times[idx - 1], times[idx]
    l0, l1 = loudness[idx - 1], loudness[idx]
    alpha = (target_time - t0) / (t1 - t0)
    return l0 + alpha * (l1 - l0)


def analyze_hihat_decay(
    drum_stem: np.ndarray,
    sample_rate: int,
    onset_time: float,
    next_onset_time: Optional[float] = None,
    window_ms: float = 150.0,
    open_threshold: float = 0.75,
) -> str:
    """
    Classify hi-hat as open or closed based on decay characteristics.
    
    Open hi-hat: slow decay, sustain stays above threshold
    Closed hi-hat: fast decay, sustain drops quickly
    
    Args:
        drum_stem: Drum audio (isolated or full)
        sample_rate: Sample rate
        onset_time: Time of the hi-hat hit
        next_onset_time: Time of next hi-hat hit (for window limit)
        window_ms: Analysis window in milliseconds
        open_threshold: If min/max ratio > this, classify as open
        
    Returns:
        "hihat_open" or "hihat_closed"
    """
    # Compute loudness curve
    times, loudness = compute_loudness_curve(drum_stem, sample_rate)
    
    # Define analysis window
    window_sec = window_ms / 1000.0
    window_end = onset_time + window_sec
    if next_onset_time is not None:
        window_end = min(window_end, next_onset_time)
    
    # Get loudness values in window
    mask = (times >= onset_time) & (times <= window_end)
    window_loudness = loudness[mask]
    
    if len(window_loudness) < 3:
        # Too short to analyze, default to closed
        return "hihat_closed"
    
    # Analyze decay
    max_loudness = window_loudness.max()
    min_loudness = window_loudness.min()
    
    if max_loudness == 0:
        return "hihat_closed"
    
    # If minimum stays above threshold of maximum, it's open (slow decay)
    ratio = min_loudness / max_loudness
    
    if ratio > open_threshold:
        return "hihat_open"
    else:
        return "hihat_closed"


def classify_cymbal_type(
    crash_stem: np.ndarray,
    ride_stem: np.ndarray,
    sample_rate: int,
    onset_time: float,
    crash_peaks: List[float],
    refractory_sec: float = 1.0,
) -> str:
    """
    Classify cymbal hit as crash or ride based on stem comparison.
    
    Uses heuristic: crash has long decay, so apply refractory period
    after crash peaks where subsequent hits are likely ride.
    
    Args:
        crash_stem: Isolated crash cymbal audio
        ride_stem: Isolated ride cymbal audio  
        sample_rate: Sample rate
        onset_time: Time of the cymbal hit
        crash_peaks: List of known crash peak times
        refractory_sec: Seconds after crash peak to classify as ride
        
    Returns:
        "crash" or "ride"
    """
    # Compute loudness curves
    crash_times, crash_loudness = compute_loudness_curve(crash_stem, sample_rate)
    ride_times, ride_loudness = compute_loudness_curve(ride_stem, sample_rate)
    
    # Check if we're in a crash refractory period
    for crash_time in crash_peaks:
        if crash_time <= onset_time <= crash_time + refractory_sec:
            # In refractory period after a crash - likely ride
            return "ride"
    
    # Compare loudness at onset time
    crash_level = get_loudness_at_time(crash_times, crash_loudness, onset_time)
    ride_level = get_loudness_at_time(ride_times, ride_loudness, onset_time)
    
    # Whichever stem is louder at this time wins
    if crash_level > ride_level * 1.5:  # Crash needs to be significantly louder
        return "crash"
    else:
        return "ride"


def find_crash_peaks(
    crash_stem: np.ndarray,
    sample_rate: int,
    threshold_ratio: float = 0.5,
    min_distance_sec: float = 0.5,
) -> List[float]:
    """
    Find peak times in crash cymbal stem.
    
    These are used to establish refractory periods for ride classification.
    """
    times, loudness = compute_loudness_curve(crash_stem, sample_rate)
    
    if len(loudness) == 0:
        return []
    
    # Normalize
    max_loud = loudness.max()
    if max_loud == 0:
        return []
    
    threshold = max_loud * threshold_ratio
    
    # Find peaks above threshold
    peaks = []
    min_distance_frames = int(min_distance_sec * sample_rate / 512)
    
    last_peak_idx = -min_distance_frames
    for i in range(1, len(loudness) - 1):
        if loudness[i] > threshold:
            if loudness[i] > loudness[i-1] and loudness[i] >= loudness[i+1]:
                if i - last_peak_idx >= min_distance_frames:
                    peaks.append(times[i])
                    last_peak_idx = i
    
    return peaks


def refine_drum_events(
    events: List[Dict[str, Any]],
    stems: Dict[str, np.ndarray],
    sample_rate: int,
    enable_hihat_split: bool = True,
    enable_cymbal_split: bool = True,
) -> List[Dict[str, Any]]:
    """
    Refine drum events by splitting hi-hat and cymbal classes.
    
    Args:
        events: List of drum events from ADTOF
        stems: Dict of isolated stems {"drums": array, "crash": array, "ride": array, ...}
        sample_rate: Sample rate of stems
        enable_hihat_split: Split hihat_closed into open/closed
        enable_cymbal_split: Split crash into crash/ride
        
    Returns:
        Refined events list with expanded classes
    """
    refined = []
    
    # Get stems (with fallbacks)
    drum_stem = stems.get("drums")
    crash_stem = stems.get("crash", stems.get("other"))
    ride_stem = stems.get("ride", stems.get("other"))
    
    # Find crash peaks for refractory period heuristic
    crash_peaks = []
    if enable_cymbal_split and crash_stem is not None:
        crash_peaks = find_crash_peaks(crash_stem, sample_rate)
        print(f"[DrumRefinement] Found {len(crash_peaks)} crash peaks for refractory periods")
    
    # Get sorted hihat times for next-onset analysis
    hihat_times = sorted([
        e["time_seconds"] for e in events 
        if e.get("instrument") in ("hihat_closed", "hihat", "hihat_open")
    ])
    
    for event in events:
        instrument = event.get("instrument", "")
        time_sec = event["time_seconds"]
        
        # Hi-hat refinement
        if enable_hihat_split and instrument in ("hihat_closed", "hihat"):
            if drum_stem is not None:
                # Find next hihat onset
                next_hihat = None
                for t in hihat_times:
                    if t > time_sec:
                        next_hihat = t
                        break
                
                # Analyze decay
                refined_instrument = analyze_hihat_decay(
                    drum_stem, sample_rate, time_sec, next_hihat
                )
                event = event.copy()
                event["instrument"] = refined_instrument
                event["original_instrument"] = instrument
                
                # Update MIDI note
                if refined_instrument == "hihat_open":
                    event["midi_note"] = 46  # Open hi-hat
                else:
                    event["midi_note"] = 42  # Closed hi-hat
        
        # Cymbal refinement (crash → crash/ride)
        elif enable_cymbal_split and instrument == "crash":
            if crash_stem is not None and ride_stem is not None:
                refined_instrument = classify_cymbal_type(
                    crash_stem, ride_stem, sample_rate, time_sec, crash_peaks
                )
                event = event.copy()
                event["instrument"] = refined_instrument
                event["original_instrument"] = instrument
                
                # Update MIDI note
                if refined_instrument == "ride":
                    event["midi_note"] = 51  # Ride cymbal
                else:
                    event["midi_note"] = 49  # Crash cymbal
        
        refined.append(event)
    
    # Report refinement stats
    if enable_hihat_split or enable_cymbal_split:
        inst_counts = {}
        for e in refined:
            inst = e.get("instrument", "unknown")
            inst_counts[inst] = inst_counts.get(inst, 0) + 1
        print(f"[DrumRefinement] Refined breakdown: {inst_counts}")
    
    return refined


class DrumRefiner:
    """
    ComfyUI-compatible drum refinement wrapper.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "events": ("DRUM_EVENTS",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "stems": ("AUDIO_STEMS",),  # If available from Demucs
                "split_hihat": ("BOOLEAN", {"default": True}),
                "split_cymbal": ("BOOLEAN", {"default": True}),
                "hihat_window_ms": ("FLOAT", {
                    "default": 150.0,
                    "min": 50.0,
                    "max": 500.0,
                    "step": 10.0,
                }),
                "hihat_open_threshold": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.5,
                    "max": 0.95,
                    "step": 0.05,
                }),
            }
        }
    
    RETURN_TYPES = ("DRUM_EVENTS",)
    RETURN_NAMES = ("refined_events",)
    FUNCTION = "refine"
    CATEGORY = "audio/Drums2Chart"
    
    def refine(
        self,
        events: List[Dict],
        audio: Dict[str, Any],
        stems: Optional[Dict[str, Any]] = None,
        split_hihat: bool = True,
        split_cymbal: bool = True,
        hihat_window_ms: float = 150.0,
        hihat_open_threshold: float = 0.75,
    ):
        """Refine drum events with enhanced classification."""
        
        sample_rate = audio.get("sample_rate", 44100)
        
        # Build stems dict
        stem_dict = {}
        
        # Use main audio as drum stem if no stems provided
        waveform = audio.get("waveform")
        if waveform is not None:
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.squeeze().cpu().numpy()
            stem_dict["drums"] = waveform
        
        # Add separated stems if provided
        if stems is not None:
            for stem_name, stem_data in stems.items():
                if isinstance(stem_data, dict):
                    stem_wf = stem_data.get("waveform")
                    if stem_wf is not None:
                        if isinstance(stem_wf, torch.Tensor):
                            stem_wf = stem_wf.squeeze().cpu().numpy()
                        stem_dict[stem_name] = stem_wf
                elif isinstance(stem_data, (np.ndarray, torch.Tensor)):
                    if isinstance(stem_data, torch.Tensor):
                        stem_data = stem_data.squeeze().cpu().numpy()
                    stem_dict[stem_name] = stem_data
        
        # Refine events
        refined = refine_drum_events(
            events=events,
            stems=stem_dict,
            sample_rate=sample_rate,
            enable_hihat_split=split_hihat,
            enable_cymbal_split=split_cymbal,
        )
        
        return (refined,)
