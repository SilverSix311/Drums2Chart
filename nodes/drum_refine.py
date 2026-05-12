"""
DrumRefine Node

Post-processing node that expands ADTOF's 5 classes into 7+ classes:
- hihat_closed → hihat_open / hihat_closed (decay analysis)
- crash → crash / ride (stem comparison)

Use AFTER DrumTranscribe and BEFORE DrumMapping.
Requires stems from StemIsolate for best cymbal classification.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple


class DrumRefine:
    """
    Refines drum transcription by splitting hi-hat and cymbal classes.
    
    Uses audio analysis heuristics:
    - Hi-hat open/closed: Analyzes decay curve after each hit
    - Crash/ride: Compares loudness in separated stems
    
    Best results when used with separated stems from Demucs.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "events": ("DRUM_EVENTS",),
                "drums_audio": ("AUDIO",),  # Isolated drum stem
            },
            "optional": {
                "crash_stem": ("AUDIO",),   # Isolated crash (from 6-stem Demucs)
                "ride_stem": ("AUDIO",),    # Isolated ride (or use 'other' stem)
                "split_hihat": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Split hi-hat into open/closed based on decay"
                }),
                "split_cymbal": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Split cymbal into crash/ride based on stem comparison"
                }),
                "hihat_window_ms": ("FLOAT", {
                    "default": 150.0,
                    "min": 50.0,
                    "max": 500.0,
                    "step": 10.0,
                    "tooltip": "Analysis window for hi-hat decay (ms)"
                }),
                "hihat_open_threshold": ("FLOAT", {
                    "default": 0.70,
                    "min": 0.4,
                    "max": 0.95,
                    "step": 0.05,
                    "tooltip": "Decay ratio threshold for open hi-hat (higher = stricter)"
                }),
                "crash_refractory_sec": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.3,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Seconds after crash peak to assume ride hits"
                }),
            }
        }
    
    RETURN_TYPES = ("DRUM_EVENTS", "STRING")
    RETURN_NAMES = ("events", "refinement_log")
    FUNCTION = "refine"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = """
    Expands ADTOF's 5-class output to 7+ classes:
    
    Before: kick, snare, tom, hihat_closed, crash
    After:  kick, snare, tom, hihat_open, hihat_closed, crash, ride
    
    Hi-hat detection: Analyzes decay curve after each hit.
    Open hi-hat has slow decay (sustain), closed has fast decay.
    
    Cymbal detection: Compares loudness in crash vs ride stems.
    Uses refractory period heuristic (crash has long decay).
    
    Connect crash_stem and ride_stem from a 6-stem Demucs
    for best cymbal classification.
    """

    def refine(
        self,
        events: List[Dict],
        drums_audio: Dict[str, Any],
        crash_stem: Optional[Dict[str, Any]] = None,
        ride_stem: Optional[Dict[str, Any]] = None,
        split_hihat: bool = True,
        split_cymbal: bool = True,
        hihat_window_ms: float = 150.0,
        hihat_open_threshold: float = 0.70,
        crash_refractory_sec: float = 1.0,
    ) -> Tuple[List[Dict], str]:
        """
        Refine drum events with enhanced classification.
        """
        from ..utils.drum_refinement import (
            refine_drum_events,
            compute_loudness_curve,
            find_crash_peaks,
        )
        
        sample_rate = drums_audio.get("sample_rate", 44100)
        log_lines = []
        
        # Extract waveforms
        stem_dict = {}
        
        # Drums stem (required)
        drums_wf = drums_audio.get("waveform")
        if drums_wf is not None:
            if isinstance(drums_wf, torch.Tensor):
                drums_wf = drums_wf.squeeze().cpu().numpy()
            stem_dict["drums"] = drums_wf
            log_lines.append(f"Drums stem: {len(drums_wf)/sample_rate:.1f}s")
        
        # Crash stem (optional)
        if crash_stem is not None:
            crash_wf = crash_stem.get("waveform")
            if crash_wf is not None:
                if isinstance(crash_wf, torch.Tensor):
                    crash_wf = crash_wf.squeeze().cpu().numpy()
                stem_dict["crash"] = crash_wf
                log_lines.append(f"Crash stem: {len(crash_wf)/sample_rate:.1f}s")
        
        # Ride stem (optional)
        if ride_stem is not None:
            ride_wf = ride_stem.get("waveform")
            if ride_wf is not None:
                if isinstance(ride_wf, torch.Tensor):
                    ride_wf = ride_wf.squeeze().cpu().numpy()
                stem_dict["ride"] = ride_wf
                log_lines.append(f"Ride stem: {len(ride_wf)/sample_rate:.1f}s")
        
        # Count original events
        orig_counts = {}
        for e in events:
            inst = e.get("instrument", "unknown")
            orig_counts[inst] = orig_counts.get(inst, 0) + 1
        log_lines.append(f"Input: {orig_counts}")
        
        # Check if we can do cymbal split
        can_split_cymbal = split_cymbal and "crash" in stem_dict and "ride" in stem_dict
        if split_cymbal and not can_split_cymbal:
            log_lines.append("Warning: No crash/ride stems, cymbal split disabled")
        
        # Refine events
        refined = refine_drum_events(
            events=events,
            stems=stem_dict,
            sample_rate=sample_rate,
            enable_hihat_split=split_hihat,
            enable_cymbal_split=can_split_cymbal,
        )
        
        # Count refined events
        refined_counts = {}
        for e in refined:
            inst = e.get("instrument", "unknown")
            refined_counts[inst] = refined_counts.get(inst, 0) + 1
        log_lines.append(f"Output: {refined_counts}")
        
        # Log changes
        if split_hihat:
            open_count = refined_counts.get("hihat_open", 0)
            closed_count = refined_counts.get("hihat_closed", 0)
            log_lines.append(f"Hi-hat: {open_count} open, {closed_count} closed")
        
        if can_split_cymbal:
            crash_count = refined_counts.get("crash", 0)
            ride_count = refined_counts.get("ride", 0)
            log_lines.append(f"Cymbals: {crash_count} crash, {ride_count} ride")
        
        log_text = "\n".join(log_lines)
        print(f"[DrumRefine] {log_text}")
        
        return (refined, log_text)


# Node registration
NODE_CLASS_MAPPINGS = {
    "DrumRefine": DrumRefine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DrumRefine": "Drum Refine (7-class)",
}
