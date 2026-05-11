"""
ADTOF-pytorch Integration

Wraps the ADTOF Frame_RNN model for use in ComfyUI nodes.
Based on: https://github.com/xavriley/ADTOF-pytorch
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

# These will be imported from the model package
# Users must: pip install adtof-pytorch
# Or we vendor it into our package

try:
    from adtof_pytorch import (
        create_frame_rnn_model,
        calculate_n_bins,
        load_pytorch_weights,
        PeakPicker,
        LABELS_5,
        FRAME_RNN_THRESHOLDS,
    )
    from adtof_pytorch.audio import create_adtof_processor
    ADTOF_AVAILABLE = True
except ImportError:
    ADTOF_AVAILABLE = False
    create_frame_rnn_model = None
    calculate_n_bins = None
    load_pytorch_weights = None
    PeakPicker = None
    LABELS_5 = None
    FRAME_RNN_THRESHOLDS = None
    create_adtof_processor = None


# GM Drum MIDI note mapping
# ADTOF outputs: [35, 38, 47, 42, 49]
# Which are: Kick(35/36), Snare(38), Tom(47), HiHat Closed(42), Crash(49)
ADTOF_TO_GM = {
    35: 36,  # Acoustic Bass Drum → Kick
    38: 38,  # Acoustic Snare
    47: 47,  # Low-Mid Tom
    42: 42,  # Closed Hi-Hat
    49: 49,  # Crash Cymbal 1
}

INSTRUMENT_NAMES = {
    35: "kick",
    38: "snare",
    47: "tom",
    42: "hihat_closed",
    49: "crash",
}


def load_adtof_model(weights_path: Optional[str] = None, device: str = "cuda") -> Dict[str, Any]:
    """
    Load ADTOF Frame_RNN model.
    
    Args:
        weights_path: Path to .pth weights file. If None, uses default packaged weights.
        device: 'cuda' or 'cpu'
        
    Returns:
        Dict with model, config, and metadata
    """
    if not ADTOF_AVAILABLE:
        raise ImportError(
            "adtof-pytorch not installed. Install with:\n"
            "  pip install adtof-pytorch\n"
            "Or from source: https://github.com/xavriley/ADTOF-pytorch"
        )
    
    # Create model
    n_bins = calculate_n_bins()
    model = create_frame_rnn_model(n_bins)
    model.eval()
    
    # Load weights
    if weights_path is not None and Path(weights_path).exists():
        model = load_pytorch_weights(model, weights_path, strict=False)
    else:
        # Try to load default packaged weights
        try:
            from importlib.resources import files
            default_weights = str(
                files("adtof_pytorch") / "data" / "adtof_frame_rnn_pytorch_weights.pth"
            )
            if Path(default_weights).exists():
                model = load_pytorch_weights(model, default_weights, strict=False)
            else:
                print("[ADTOF] Warning: No weights loaded. Model will produce garbage.")
        except Exception as e:
            print(f"[ADTOF] Warning: Could not load default weights: {e}")
    
    # Move to device
    model.to(device)
    
    config = {
        "sample_rate": 44100,
        "fps": 100,  # Frames per second of model activations
        "n_bins": n_bins,
        "labels": list(LABELS_5),
        "thresholds": list(FRAME_RNN_THRESHOLDS),
        "instrument_names": INSTRUMENT_NAMES,
    }
    
    return {
        "model": model,
        "config": config,
        "n_bins": n_bins,
    }


def transcribe_adtof(
    model_obj: torch.nn.Module,
    config: Dict[str, Any],
    waveform: torch.Tensor,
    sample_rate: int,
    sensitivity: float = 0.5,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Run ADTOF transcription on audio.
    
    Args:
        model_obj: Loaded ADTOF model
        config: Model config dict
        waveform: Audio tensor [1, 1, samples] (mono)
        sample_rate: Sample rate (will resample to 44.1kHz)
        sensitivity: Detection sensitivity 0-1 (higher = more sensitive)
        device: 'cuda' or 'cpu'
        
    Returns:
        List of drum events with time, instrument, velocity, midi_note
    """
    if not ADTOF_AVAILABLE:
        raise ImportError("adtof-pytorch not installed")
    
    # Resample to 44.1kHz if needed
    if sample_rate != 44100:
        import torchaudio.transforms as T
        resampler = T.Resample(sample_rate, 44100)
        waveform = resampler(waveform)
    
    # Convert to numpy for audio processor
    audio_np = waveform.squeeze().cpu().numpy()
    
    # Ensure mono and 1D
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)
    
    # Create processor and compute spectrogram using its methods
    processor = create_adtof_processor()
    
    # The processor has compute_stft and apply_filterbank methods
    # process_audio takes a file path, but we have raw audio, so use the components directly
    stft = processor.compute_stft(audio_np)  # [freq_bins, time]
    filtered = processor.apply_filterbank(stft)  # [n_bins, time]
    
    # Transpose to [time, freq_bins] and add channel dim for model
    mel_spec = filtered.T[:, :, np.newaxis]  # [time, freq_bins, 1]
    
    # Add batch dim and convert to tensor
    x = torch.from_numpy(mel_spec).unsqueeze(0).float().to(device)  # [1, time, freq_bins, 1]
    
    # Run model
    with torch.no_grad():
        activations = model_obj(x).cpu().numpy()  # [1, time, 5 classes]
    
    # Adjust thresholds based on sensitivity
    # sensitivity 0.5 = default thresholds
    # sensitivity 1.0 = very sensitive (low thresholds)
    # sensitivity 0.0 = very insensitive (high thresholds)
    base_thresholds = config.get("thresholds", FRAME_RNN_THRESHOLDS)
    adjusted_thresholds = [
        max(0.05, min(0.95, t * (1.5 - sensitivity)))
        for t in base_thresholds
    ]
    
    # Peak picking
    fps = config.get("fps", 100)
    picker = PeakPicker(thresholds=adjusted_thresholds, fps=fps)
    labels = config.get("labels", LABELS_5)
    peaks_dict_list = picker.pick(activations, labels=labels, label_offset=0)
    peaks_dict = peaks_dict_list[0]  # First (and only) batch item
    
    # Convert to event list
    events = []
    for midi_note, times in peaks_dict.items():
        instrument = INSTRUMENT_NAMES.get(midi_note, f"unknown_{midi_note}")
        gm_note = ADTOF_TO_GM.get(midi_note, midi_note)
        
        for time_sec in times:
            # Estimate velocity from activation strength at that time
            frame_idx = int(time_sec * fps)
            class_idx = labels.index(midi_note) if midi_note in labels else 0
            
            if frame_idx < activations.shape[1]:
                activation_value = activations[0, frame_idx, class_idx]
                velocity = int(np.clip(activation_value * 127, 20, 127))
            else:
                velocity = 80  # Default
            
            events.append({
                "time_seconds": float(time_sec),
                "instrument": instrument,
                "velocity": velocity,
                "midi_note": gm_note,
                "confidence": float(activation_value) if frame_idx < activations.shape[1] else 0.5,
            })
    
    # Sort by time
    events.sort(key=lambda e: e["time_seconds"])
    
    return events


def get_adtof_info() -> Dict[str, Any]:
    """Get ADTOF model information"""
    return {
        "name": "ADTOF Frame_RNN",
        "description": "Drum transcription trained on 114 hours of rhythm game data",
        "classes": 5,
        "instruments": ["kick", "snare", "tom", "hihat_closed", "crash"],
        "sample_rate": 44100,
        "fps": 100,
        "available": ADTOF_AVAILABLE,
        "paper": "https://doi.org/10.3390/signals4040042",
    }
