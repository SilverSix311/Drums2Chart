"""
DrumTranscribe Node

Converts audio (ideally isolated drum stem) to MIDI using AI transcription.
Primary model: ADTOF (Automatic Drum Transcription using Onset-detection and Filter-bank features)
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any

# TODO: Import ADTOF or alternative model
# from models.adtof import ADTOFModel


class DrumTranscribe:
    """
    AI-powered drum transcription node.
    
    Takes an audio tensor (preferably isolated drums from Demucs) and outputs
    MIDI note events for drum hits.
    
    Supported models:
    - ADTOF (default, best accuracy)
    - Omnizart (fallback, easier setup)
    - Onsets & Frames (Google/Magenta baseline)
    """
    
    MODELS = ["adtof", "omnizart", "onsets_frames"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),  # ComfyUI audio tensor format
                "model": (cls.MODELS, {"default": "adtof"}),
                "sensitivity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Detection sensitivity - higher = more notes detected (may include false positives)"
                }),
            },
            "optional": {
                "velocity_threshold": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 127,
                    "step": 1,
                    "tooltip": "Minimum MIDI velocity to include (filters quiet ghost notes)"
                }),
            }
        }
    
    RETURN_TYPES = ("MIDI_DATA", "DRUM_EVENTS")
    RETURN_NAMES = ("midi", "events")
    FUNCTION = "transcribe"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = """
    Transcribes drum audio to MIDI using AI models.
    
    Best results with isolated drum stems (use AudioSeparateDemucs first).
    
    Output:
    - midi: MIDI data structure for downstream processing
    - events: Raw drum event list with timestamps, instruments, velocities
    """

    def transcribe(
        self,
        audio: Dict[str, Any],
        model: str = "adtof",
        sensitivity: float = 0.5,
        velocity_threshold: int = 20,
    ) -> Tuple[Dict, list]:
        """
        Main transcription function.
        
        Args:
            audio: ComfyUI audio dict with 'waveform' tensor and 'sample_rate'
            model: Which transcription model to use
            sensitivity: Detection threshold (0-1)
            velocity_threshold: Minimum velocity to keep
            
        Returns:
            Tuple of (midi_data dict, drum_events list)
        """
        # Extract audio data
        waveform = audio["waveform"]  # Shape: [batch, channels, samples]
        sample_rate = audio["sample_rate"]
        
        # Convert to mono if stereo
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        
        # TODO: Implement actual model inference
        # For now, return placeholder structure
        
        if model == "adtof":
            events = self._transcribe_adtof(waveform, sample_rate, sensitivity)
        elif model == "omnizart":
            events = self._transcribe_omnizart(waveform, sample_rate, sensitivity)
        else:
            events = self._transcribe_onsets_frames(waveform, sample_rate, sensitivity)
        
        # Filter by velocity
        events = [e for e in events if e["velocity"] >= velocity_threshold]
        
        # Convert to MIDI data structure
        midi_data = self._events_to_midi(events, sample_rate)
        
        return (midi_data, events)
    
    def _transcribe_adtof(self, waveform: torch.Tensor, sample_rate: int, sensitivity: float) -> list:
        """ADTOF model inference"""
        # TODO: Implement ADTOF integration
        # Reference: https://github.com/MZehren/ADTOF
        raise NotImplementedError("ADTOF model integration pending")
    
    def _transcribe_omnizart(self, waveform: torch.Tensor, sample_rate: int, sensitivity: float) -> list:
        """Omnizart model inference"""
        # TODO: Implement Omnizart integration
        # omnizart drum transcribe -> MIDI
        raise NotImplementedError("Omnizart model integration pending")
    
    def _transcribe_onsets_frames(self, waveform: torch.Tensor, sample_rate: int, sensitivity: float) -> list:
        """Google Magenta Onsets & Frames inference"""
        # TODO: Implement O&F integration
        raise NotImplementedError("Onsets & Frames model integration pending")
    
    def _events_to_midi(self, events: list, sample_rate: int) -> Dict:
        """
        Convert drum events to MIDI data structure.
        
        Event format:
        {
            "time_seconds": float,
            "instrument": str (kick/snare/hihat_closed/etc),
            "velocity": int (1-127),
            "midi_note": int
        }
        """
        # Standard GM drum map
        DRUM_MIDI_MAP = {
            "kick": 36,
            "snare": 38,
            "hihat_closed": 42,
            "hihat_open": 46,
            "hihat_pedal": 44,
            "tom_high": 50,
            "tom_mid": 47,
            "tom_low": 45,
            "tom_floor": 43,
            "crash": 49,
            "ride": 51,
            "ride_bell": 53,
        }
        
        midi_data = {
            "ticks_per_beat": 480,  # Standard MIDI resolution
            "tempo": 120,  # Will be updated by tempo detection
            "tracks": [{
                "name": "Drums",
                "channel": 9,  # GM drum channel
                "events": []
            }]
        }
        
        for event in events:
            midi_note = event.get("midi_note") or DRUM_MIDI_MAP.get(event["instrument"], 38)
            midi_data["tracks"][0]["events"].append({
                "type": "note_on",
                "time": event["time_seconds"],
                "note": midi_note,
                "velocity": event["velocity"],
                "duration": 0.05,  # Drums are typically short
            })
        
        return midi_data
