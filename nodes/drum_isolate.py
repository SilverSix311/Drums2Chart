"""
DrumIsolate Node

Isolates drum stem from mixed audio using AI source separation (Demucs).
"""

import torch
import torchaudio
from typing import Dict, Any, Tuple

# Check for torchaudio's demucs
try:
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False


class DrumIsolate:
    """
    Isolates drums from mixed audio using Hybrid Demucs.
    
    Uses torchaudio's built-in Demucs model for high-quality stem separation.
    Outputs isolated drums and optionally the backing track (everything minus drums).
    
    Models available:
    - htdemucs (default): 4 stems - drums, bass, vocals, other
    - htdemucs_ft: Fine-tuned version, slightly better quality
    """
    
    MODELS = ["htdemucs", "htdemucs_ft", "htdemucs_6s"]
    OUTPUT_MODES = ["drums_only", "drums_and_backing", "all_stems"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (cls.MODELS, {
                    "default": "htdemucs",
                    "tooltip": "Demucs model variant. htdemucs_ft is higher quality but slower."
                }),
                "output_mode": (cls.OUTPUT_MODES, {
                    "default": "drums_only",
                    "tooltip": "What to output: just drums, drums + backing, or all stems"
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Processing device"
                }),
                "shifts": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Number of random shifts for better quality (more = slower but better)"
                }),
                "overlap": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": "Overlap between chunks (higher = better quality at boundaries)"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("drums", "backing", "bass", "vocals", "other")
    FUNCTION = "isolate"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = """
    Isolates drums from mixed audio using Hybrid Demucs.
    
    Outputs:
    - drums: Isolated drum track
    - backing: Everything except drums (bass + vocals + other)
    - bass: Isolated bass (if output_mode includes it)
    - vocals: Isolated vocals (if output_mode includes it)
    - other: Everything else - guitars, synths, etc.
    
    For best quality with complex audio, increase 'shifts' (but it's slower).
    """

    def __init__(self):
        self._model = None
        self._model_name = None
    
    def _load_model(self, model_name: str, device: str):
        """Load Demucs model (cached)"""
        if self._model is not None and self._model_name == model_name:
            return self._model
        
        if not DEMUCS_AVAILABLE:
            raise ImportError(
                "torchaudio Demucs not available. "
                "Update torchaudio: pip install --upgrade torchaudio"
            )
        
        print(f"[Drums2Chart] Loading Demucs model: {model_name}")
        
        # Use torchaudio's bundled model
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        model = bundle.get_model()
        model.to(device)
        model.eval()
        
        self._model = model
        self._model_name = model_name
        self._sample_rate = bundle.sample_rate  # 44100
        
        return model
    
    def isolate(
        self,
        audio: Dict[str, Any],
        model: str = "htdemucs",
        output_mode: str = "drums_only",
        device: str = "auto",
        shifts: int = 1,
        overlap: float = 0.25,
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Isolate drums from audio.
        
        Returns tuple of (drums, backing, bass, vocals, other) audio dicts.
        Unused outputs based on output_mode will have zeroed audio.
        """
        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Extract audio
        waveform = audio["waveform"]  # [batch, channels, samples]
        sample_rate = audio["sample_rate"]
        
        # Remove batch dim if present, ensure [channels, samples]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        
        # Resample to 44.1kHz if needed (Demucs requirement)
        if sample_rate != 44100:
            print(f"[Drums2Chart] Resampling from {sample_rate}Hz to 44100Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 44100)
            waveform = resampler(waveform)
            sample_rate = 44100
        
        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        
        # Load model
        demucs = self._load_model(model, device)
        waveform = waveform.to(device)
        
        # Add batch dim for model: [1, channels, samples]
        waveform_batch = waveform.unsqueeze(0)
        
        print(f"[Drums2Chart] Separating audio ({waveform.shape[1] / sample_rate:.1f}s) with {shifts} shift(s)...")
        
        # Run separation
        with torch.no_grad():
            # Demucs outputs: [batch, sources, channels, samples]
            # Sources: drums, bass, other, vocals (in that order for HDEMUCS)
            sources = self._separate_with_shifts(demucs, waveform_batch, shifts, overlap)
        
        # Extract stems - order is: drums, bass, other, vocals
        drums = sources[0, 0]      # [channels, samples]
        bass = sources[0, 1]
        other = sources[0, 2]
        vocals = sources[0, 3]
        
        # Create backing track (everything except drums)
        backing = bass + other + vocals
        
        # Move to CPU and create output dicts
        def make_audio_dict(tensor: torch.Tensor) -> Dict[str, Any]:
            return {
                "waveform": tensor.unsqueeze(0).cpu(),  # Add batch dim back
                "sample_rate": sample_rate,
            }
        
        drums_out = make_audio_dict(drums)
        backing_out = make_audio_dict(backing)
        bass_out = make_audio_dict(bass)
        vocals_out = make_audio_dict(vocals)
        other_out = make_audio_dict(other)
        
        print(f"[Drums2Chart] Separation complete!")
        
        return (drums_out, backing_out, bass_out, vocals_out, other_out)
    
    def _separate_with_shifts(
        self,
        model: torch.nn.Module,
        waveform: torch.Tensor,
        shifts: int,
        overlap: float,
    ) -> torch.Tensor:
        """
        Apply separation with random shifts for better quality.
        
        Multiple shifts help reduce artifacts at chunk boundaries.
        """
        if shifts == 1:
            return model(waveform)
        
        # Multiple shifts - average the results
        results = []
        length = waveform.shape[-1]
        
        for i in range(shifts):
            # Random shift
            shift = int(torch.randint(0, length, (1,)).item()) if i > 0 else 0
            
            # Shift waveform
            shifted = torch.roll(waveform, shift, dims=-1)
            
            # Separate
            separated = model(shifted)
            
            # Shift back
            separated = torch.roll(separated, -shift, dims=-1)
            
            results.append(separated)
        
        # Average all shifts
        return torch.stack(results).mean(dim=0)


class DrumIsolateSimple:
    """
    Simplified drum isolation - just drums output.
    
    For workflows that only need the drum stem.
    Uses same Demucs backend but simpler interface.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "quality": (["fast", "balanced", "high"], {
                    "default": "balanced",
                    "tooltip": "Quality preset: fast (1 shift), balanced (2), high (5)"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("drums",)
    FUNCTION = "isolate"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = "Simple drum isolation - input audio, output drums."

    def __init__(self):
        self._isolator = DrumIsolate()
    
    def isolate(
        self,
        audio: Dict[str, Any],
        quality: str = "balanced",
    ) -> Tuple[Dict]:
        """Simple isolation - just returns drums"""
        shifts = {"fast": 1, "balanced": 2, "high": 5}[quality]
        
        drums, _, _, _, _ = self._isolator.isolate(
            audio=audio,
            model="htdemucs",
            output_mode="drums_only",
            device="auto",
            shifts=shifts,
            overlap=0.25,
        )
        
        return (drums,)
