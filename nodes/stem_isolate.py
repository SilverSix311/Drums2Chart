"""
StemIsolate Node

Separates mixed audio into individual stems using Demucs.
Supports 4-stem and 6-stem models for maximum flexibility.
"""

import torch
import torchaudio
import os
from typing import Dict, Any, Tuple, Optional


class StemIsolate:
    """
    Separate mixed audio into individual instrument stems.
    
    Uses Demucs (Hybrid Transformer Demucs) for high-quality separation.
    
    Models:
    - htdemucs: 4 stems (drums, bass, other, vocals)
    - htdemucs_ft: 4 stems, fine-tuned (better quality)
    - htdemucs_6s: 6 stems (drums, bass, guitar, piano, other, vocals)
    
    The 6-stem model gives you dedicated guitar and keys/piano tracks!
    """
    
    MODELS = [
        "htdemucs",      # 4 stems: drums, bass, other, vocals
        "htdemucs_ft",   # 4 stems, fine-tuned
        "htdemucs_6s",   # 6 stems: drums, bass, guitar, piano, other, vocals
    ]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (cls.MODELS, {
                    "default": "htdemucs_6s",
                    "tooltip": "htdemucs_6s gives guitar + piano stems; htdemucs_ft is highest quality 4-stem"
                }),
            },
            "optional": {
                "shifts": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "tooltip": "Random shifts for better quality (higher = slower but better)"
                }),
                "overlap": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": "Overlap between chunks"
                }),
                "split": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Split into chunks (required for long audio)"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Processing device"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("drums", "bass", "guitar", "vocals", "keys", "other", "backing")
    FUNCTION = "separate"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = """
    Separate audio into stems using Demucs.
    
    Outputs:
    - drums: Drum kit (kick, snare, toms, cymbals)
    - bass: Bass guitar/synth bass
    - guitar: Guitar (rhythm + lead combined)
    - vocals: Vocals/voice
    - keys: Piano/keyboards (6-stem model only)
    - other: Everything else not categorized
    - backing: Everything except drums (for practice tracks)
    
    Use htdemucs_6s for guitar + keys separation.
    Use htdemucs_ft for highest quality 4-stem separation.
    """

    def separate(
        self,
        audio: Dict[str, Any],
        model: str = "htdemucs_6s",
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        device: str = "auto",
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict]:
        """Separate audio into stems"""
        
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        
        # Get device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[Drums2Chart] Loading Demucs model: {model}")
        separator = get_model(model)
        separator = separator.to(device)
        separator.eval()
        
        # Get audio data
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Ensure correct format [batch, channels, samples]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        
        # Resample if needed (Demucs expects 44100)
        if sample_rate != separator.samplerate:
            print(f"[Drums2Chart] Resampling {sample_rate} → {separator.samplerate}")
            resampler = torchaudio.transforms.Resample(sample_rate, separator.samplerate)
            waveform = resampler(waveform)
            sample_rate = separator.samplerate
        
        # Move to device
        waveform = waveform.to(device)
        
        # Ensure stereo
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        
        print(f"[Drums2Chart] Separating audio ({waveform.shape[-1]/sample_rate:.1f}s)...")
        
        # Apply model
        with torch.no_grad():
            sources = apply_model(
                separator,
                waveform,
                shifts=shifts,
                overlap=overlap,
                split=split,
                progress=True,
            )
        
        # Get stem names from model
        stem_names = separator.sources  # e.g., ['drums', 'bass', 'other', 'vocals'] or 6-stem
        print(f"[Drums2Chart] Model stems: {stem_names}")
        
        # Build output dict
        stems = {}
        for i, name in enumerate(stem_names):
            stems[name] = sources[0, i:i+1, :, :]  # Keep batch dim
        
        # Create output AUDIO dicts
        def make_audio(tensor: torch.Tensor) -> Dict[str, Any]:
            return {
                "waveform": tensor.cpu(),
                "sample_rate": sample_rate,
            }
        
        # Map to our standardized outputs
        drums = make_audio(stems.get("drums", torch.zeros_like(waveform)))
        bass = make_audio(stems.get("bass", torch.zeros_like(waveform)))
        vocals = make_audio(stems.get("vocals", torch.zeros_like(waveform)))
        other = make_audio(stems.get("other", torch.zeros_like(waveform)))
        
        # Guitar and keys only in 6-stem model
        if "guitar" in stems:
            guitar = make_audio(stems["guitar"])
        else:
            # For 4-stem model, "other" contains guitars
            guitar = make_audio(stems.get("other", torch.zeros_like(waveform)))
        
        if "piano" in stems:
            keys = make_audio(stems["piano"])
        else:
            # No separate keys in 4-stem model
            keys = make_audio(torch.zeros_like(waveform))
        
        # Create backing track (everything except drums)
        backing_sources = []
        for name in stem_names:
            if name != "drums":
                backing_sources.append(stems[name])
        
        if backing_sources:
            backing_waveform = sum(backing_sources)
            backing = make_audio(backing_waveform)
        else:
            backing = make_audio(torch.zeros_like(waveform))
        
        print(f"[Drums2Chart] Separation complete!")
        
        return (drums, bass, guitar, vocals, keys, other, backing)


class StemIsolateSimple:
    """
    Simplified stem separation - just get drums and backing.
    
    For when you just need drums isolated quickly.
    """
    
    QUALITY = ["fast", "balanced", "high"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "quality": (cls.QUALITY, {
                    "default": "balanced",
                    "tooltip": "fast=0 shifts, balanced=1, high=5"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("drums", "backing")
    FUNCTION = "separate"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = "Quick drum isolation. Outputs drums and backing track."

    def separate(
        self,
        audio: Dict[str, Any],
        quality: str = "balanced",
    ) -> Tuple[Dict, Dict]:
        """Quick separation - just drums and backing"""
        
        shifts = {"fast": 0, "balanced": 1, "high": 5}[quality]
        
        # Use full separator
        separator = StemIsolate()
        drums, bass, guitar, vocals, keys, other, backing = separator.separate(
            audio=audio,
            model="htdemucs_ft",  # Best quality 4-stem for speed
            shifts=shifts,
            overlap=0.25,
            split=True,
            device="auto",
        )
        
        return (drums, backing)
