"""
DrumTranscribe Node

Converts audio (ideally isolated drum stem) to MIDI using AI transcription.
Primary model: ADTOF (Automatic Drum Transcription using Onset-detection and Filter-bank features)
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional


class DrumTranscribe:
    """
    AI-powered drum transcription node.
    
    Takes an audio tensor (preferably isolated drums from Demucs) and a loaded
    drum model, outputs MIDI note events for drum hits.
    
    Use with LoadDrumModel node - load the model once, reuse for multiple tracks.
    
    Supported models (via LoadDrumModel):
    - ADTOF (best accuracy for real music)
    - Omnizart (good baseline, easy setup)
    - Onsets & Frames (Google/Magenta)
    - Custom ONNX models
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),  # ComfyUI audio tensor format
                "model": ("DRUM_MODEL",),  # From LoadDrumModel node
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
                "instruments": ("STRING", {
                    "default": "kick,snare,hihat,tom,cymbal",
                    "tooltip": "Comma-separated list of instruments to detect (model-dependent)"
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
        model: Dict[str, Any],
        sensitivity: float = 0.5,
        velocity_threshold: int = 20,
        instruments: str = "kick,snare,hihat,tom,cymbal",
    ) -> Tuple[Dict, list]:
        """
        Main transcription function.
        
        Args:
            audio: ComfyUI audio dict with 'waveform' tensor and 'sample_rate'
            model: Loaded model from LoadDrumModel node
            sensitivity: Detection threshold (0-1)
            velocity_threshold: Minimum velocity to keep
            instruments: Comma-separated instruments to detect
            
        Returns:
            Tuple of (midi_data dict, drum_events list)
        """
        # Extract audio data
        waveform = audio["waveform"]  # Shape: [batch, channels, samples]
        sample_rate = audio["sample_rate"]
        
        # Convert to mono if stereo
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        
        # Get model type and object
        model_type = model.get("model_type", "adtof")
        model_obj = model.get("model")
        model_config = model.get("config", {})
        device = model.get("device", "cpu")
        
        # Parse instrument filter
        instrument_list = [i.strip().lower() for i in instruments.split(",")]
        
        # Resample if needed
        target_sr = model_config.get("sample_rate", 44100)
        if sample_rate != target_sr:
            waveform = self._resample(waveform, sample_rate, target_sr)
            sample_rate = target_sr
        
        # Move to device
        waveform = waveform.to(device)
        
        # Run inference based on model type
        print(f"[Drums2Chart] Transcribing with {model_type} (sensitivity={sensitivity})")
        
        if model_type == "adtof":
            events = self._transcribe_adtof(model_obj, model_config, waveform, sample_rate, sensitivity, instrument_list)
        elif model_type == "omnizart":
            events = self._transcribe_omnizart(model_obj, model_config, waveform, sample_rate, sensitivity, instrument_list)
        elif model_type == "onsets_frames":
            events = self._transcribe_onsets_frames(model_obj, model_config, waveform, sample_rate, sensitivity, instrument_list)
        elif model_type == "onnx_generic":
            events = self._transcribe_onnx(model_obj, model_config, waveform, sample_rate, sensitivity, instrument_list)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Filter by velocity
        events = [e for e in events if e["velocity"] >= velocity_threshold]
        
        print(f"[Drums2Chart] Detected {len(events)} drum hits")
        
        # Convert to MIDI data structure
        midi_data = self._events_to_midi(events, sample_rate)
        
        return (midi_data, events)
    
    def _resample(self, waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate"""
        import torchaudio.transforms as T
        resampler = T.Resample(orig_sr, target_sr)
        return resampler(waveform)
    
    def _transcribe_adtof(
        self, 
        model_obj: Any, 
        config: Dict, 
        waveform: torch.Tensor, 
        sample_rate: int, 
        sensitivity: float,
        instruments: list
    ) -> list:
        """ADTOF model inference"""
        # TODO: Implement actual ADTOF inference
        # Reference: https://github.com/MZehren/ADTOF
        #
        # ADTOF uses a CRNN architecture:
        # 1. Compute mel spectrogram
        # 2. Run through CNN encoder
        # 3. Run through RNN for temporal modeling
        # 4. Apply threshold to get onsets
        #
        # Example (pseudocode):
        # mel = compute_mel_spectrogram(waveform, sample_rate)
        # with torch.no_grad():
        #     activations = model_obj(mel)
        # onsets = peak_pick(activations, threshold=1-sensitivity)
        
        if model_obj.get("_placeholder"):
            raise NotImplementedError(
                "ADTOF model integration pending. "
                "Please download ADTOF model and place in models/drums2chart/"
            )
        
        return []
    
    def _transcribe_omnizart(
        self,
        model_obj: Any,
        config: Dict,
        waveform: torch.Tensor,
        sample_rate: int,
        sensitivity: float,
        instruments: list
    ) -> list:
        """Omnizart model inference"""
        # TODO: Implement Omnizart integration
        # omnizart has its own CLI and Python API
        #
        # from omnizart.drum import app as drum_app
        # result = drum_app.transcribe(audio_path)
        
        if model_obj.get("_placeholder"):
            raise NotImplementedError(
                "Omnizart model integration pending. "
                "Install with: pip install omnizart && omnizart download-checkpoints"
            )
        
        return []
    
    def _transcribe_onsets_frames(
        self,
        model_obj: Any,
        config: Dict,
        waveform: torch.Tensor,
        sample_rate: int,
        sensitivity: float,
        instruments: list
    ) -> list:
        """Google Magenta Onsets & Frames inference"""
        # TODO: Implement O&F integration
        
        if model_obj.get("_placeholder"):
            raise NotImplementedError(
                "Onsets & Frames model integration pending. "
                "Download from Magenta project."
            )
        
        return []
    
    def _transcribe_onnx(
        self,
        model_obj: Any,  # onnxruntime InferenceSession
        config: Dict,
        waveform: torch.Tensor,
        sample_rate: int,
        sensitivity: float,
        instruments: list
    ) -> list:
        """Generic ONNX model inference"""
        import numpy as np
        
        # Prepare input
        input_name = config["input_names"][0]
        audio_np = waveform.squeeze().cpu().numpy()
        
        # Run inference
        outputs = model_obj.run(None, {input_name: audio_np.astype(np.float32)})
        
        # Parse outputs (model-specific)
        # This is generic - specific models may need custom parsing
        activations = outputs[0]
        
        # Simple peak picking
        events = []
        hop_length = config.get("hop_length", 512)
        threshold = 1 - sensitivity
        
        for i, frame in enumerate(activations):
            for inst_idx, activation in enumerate(frame):
                if activation > threshold:
                    time_sec = (i * hop_length) / sample_rate
                    events.append({
                        "time_seconds": time_sec,
                        "instrument": f"drum_{inst_idx}",
                        "velocity": int(activation * 127),
                        "midi_note": 36 + inst_idx,  # Map to GM drum notes
                    })
        
        return events
    
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
