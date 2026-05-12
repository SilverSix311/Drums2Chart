"""
Onsets & Frames Drums Integration

Google Magenta's drum transcription model trained on E-GMD dataset.
444 hours of drum performances, 43 drum kits.

Detects: kick, snare, hi-hat (open/closed), toms, ride, crash, etc.
With velocity estimation!

Requires: pip install magenta
Checkpoint: Downloaded automatically from Google Cloud Storage
"""

import os
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from pathlib import Path

# E-GMD drum mapping (MIDI note -> instrument name)
# Based on Roland TD-11 mapping used in E-GMD
EGMD_DRUM_MAPPING = {
    36: "kick",
    38: "snare",
    40: "snare_rimshot",
    37: "snare_xstick",
    48: "tom_high",
    47: "tom_mid",
    45: "tom_low",
    43: "tom_floor",
    46: "hihat_open",
    42: "hihat_closed",
    44: "hihat_pedal",
    49: "crash",
    57: "crash_2",
    51: "ride",
    59: "ride_2",
    53: "ride_bell",
    55: "splash",
    52: "china",
}

# Reverse mapping
INSTRUMENT_TO_MIDI = {v: k for k, v in EGMD_DRUM_MAPPING.items()}

# Check if magenta is available
try:
    import tensorflow as tf
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    from magenta.models.onsets_frames_transcription import configs
    from magenta.models.onsets_frames_transcription import data
    from magenta.models.onsets_frames_transcription import train_util
    from magenta.music import audio_io
    from magenta.music import sequences_lib
    import note_seq
    
    MAGENTA_AVAILABLE = True
except ImportError as e:
    MAGENTA_AVAILABLE = False
    MAGENTA_IMPORT_ERROR = str(e)


def check_magenta_available() -> bool:
    """Check if Magenta is installed"""
    return MAGENTA_AVAILABLE


def get_oaf_drums_info() -> Dict[str, Any]:
    """Get OaF Drums model information"""
    return {
        "name": "Onsets & Frames Drums (E-GMD)",
        "description": "Drum transcription trained on 444 hours of human performances",
        "classes": len(EGMD_DRUM_MAPPING),
        "instruments": list(EGMD_DRUM_MAPPING.values()),
        "has_velocity": True,
        "sample_rate": 16000,
        "available": MAGENTA_AVAILABLE,
        "paper": "https://arxiv.org/abs/1906.01431",
    }


def download_oaf_drums_checkpoint(model_dir: str) -> str:
    """
    Download the E-GMD trained checkpoint if not present.
    
    Returns path to checkpoint directory.
    """
    checkpoint_dir = Path(model_dir) / "oaf_drums_egmd"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    ckpt_file = checkpoint_dir / "model.ckpt-200000.index"
    if ckpt_file.exists():
        print(f"[OaF Drums] Checkpoint already exists at {checkpoint_dir}")
        return str(checkpoint_dir)
    
    print(f"[OaF Drums] Downloading E-GMD checkpoint to {checkpoint_dir}...")
    
    # GCS bucket for E-GMD trained model
    # Note: This URL may need to be updated based on actual availability
    import urllib.request
    import tarfile
    
    # The checkpoint should be available from Magenta's releases
    # For now, we'll use the Colab-style download
    gcs_base = "https://storage.googleapis.com/magentadata/models/onsets_frames_transcription"
    checkpoint_name = "e-gmd_checkpoint"
    
    try:
        tar_path = checkpoint_dir / "checkpoint.tar.gz"
        url = f"{gcs_base}/{checkpoint_name}.tar.gz"
        
        print(f"[OaF Drums] Fetching from {url}")
        urllib.request.urlretrieve(url, tar_path)
        
        # Extract
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(checkpoint_dir)
        
        # Clean up
        tar_path.unlink()
        print(f"[OaF Drums] Checkpoint downloaded successfully")
        
    except Exception as e:
        print(f"[OaF Drums] Auto-download failed: {e}")
        print("[OaF Drums] Please download manually from:")
        print("  https://magenta.withgoogle.com/oaf-drums")
        print(f"  Extract to: {checkpoint_dir}")
        raise
    
    return str(checkpoint_dir)


def load_oaf_drums_model(
    checkpoint_path: Optional[str] = None,
    model_dir: str = "./models/drums2chart",
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Load the Onsets & Frames Drums model.
    
    Args:
        checkpoint_path: Path to specific checkpoint, or None to auto-download
        model_dir: Directory to store/find checkpoints
        device: 'cuda' or 'cpu' (TF will use GPU if available)
        
    Returns:
        Dict with model, config, and metadata
    """
    if not MAGENTA_AVAILABLE:
        raise ImportError(
            f"Magenta not installed. Install with:\n"
            f"  pip install magenta\n"
            f"Import error was: {MAGENTA_IMPORT_ERROR}"
        )
    
    # Configure TF to use GPU if requested
    if device == "cuda":
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[OaF Drums] Using GPU: {gpus[0].name}")
            except RuntimeError as e:
                print(f"[OaF Drums] GPU config error: {e}")
    else:
        tf.config.set_visible_devices([], 'GPU')
        print("[OaF Drums] Using CPU")
    
    # Get checkpoint path
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        checkpoint_path = download_oaf_drums_checkpoint(model_dir)
    
    # Load config for drums
    config = configs.CONFIG_MAP['drums']
    
    # Find checkpoint file
    ckpt_dir = Path(checkpoint_path)
    ckpt_files = list(ckpt_dir.glob("model.ckpt-*.index"))
    if not ckpt_files:
        ckpt_files = list(ckpt_dir.glob("*.index"))
    
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    
    # Use the latest checkpoint
    ckpt_path = str(ckpt_files[0]).replace(".index", "")
    print(f"[OaF Drums] Loading checkpoint: {ckpt_path}")
    
    # Build the model
    hparams = config.hparams
    hparams.batch_size = 1
    
    # Create estimator
    model_fn = train_util.create_model_fn(
        model_fn=config.model_fn,
        hparams=hparams,
        model_dir=checkpoint_path,
    )
    
    # We'll use the estimator pattern for inference
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=checkpoint_path,
    )
    
    return {
        "estimator": estimator,
        "config": config,
        "hparams": hparams,
        "checkpoint_path": checkpoint_path,
        "sample_rate": 16000,
        "drum_mapping": EGMD_DRUM_MAPPING,
    }


def transcribe_oaf_drums(
    model_dict: Dict[str, Any],
    waveform: torch.Tensor,
    sample_rate: int,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Run OaF Drums transcription on audio.
    
    Args:
        model_dict: Loaded model dictionary
        waveform: Audio tensor [batch, channels, samples] or [channels, samples]
        sample_rate: Sample rate of input audio
        onset_threshold: Threshold for onset detection (0-1)
        frame_threshold: Threshold for frame detection (0-1)
        
    Returns:
        List of drum events with time, instrument, velocity, midi_note
    """
    if not MAGENTA_AVAILABLE:
        raise ImportError("Magenta not installed")
    
    # Convert to numpy
    audio_np = waveform.squeeze().cpu().numpy()
    
    # Ensure mono
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)
    
    # Resample to 16kHz if needed (OaF uses 16kHz)
    target_sr = model_dict["sample_rate"]
    if sample_rate != target_sr:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=target_sr)
    
    # Normalize
    audio_np = audio_np.astype(np.float32)
    if np.abs(audio_np).max() > 1.0:
        audio_np = audio_np / np.abs(audio_np).max()
    
    print(f"[OaF Drums] Transcribing {len(audio_np)/target_sr:.1f}s of audio...")
    
    # Use Magenta's transcription pipeline
    config = model_dict["config"]
    hparams = model_dict["hparams"]
    checkpoint_path = model_dict["checkpoint_path"]
    
    # Create a NoteSequence from audio using the transcription
    # This uses the full Magenta pipeline
    from magenta.models.onsets_frames_transcription import infer_util
    
    sequence = infer_util.transcribe_audio(
        model_dir=checkpoint_path,
        config=config,
        hparams=hparams,
        audio=audio_np,
        sample_rate=target_sr,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )
    
    # Convert NoteSequence to our event format
    events = []
    drum_mapping = model_dict["drum_mapping"]
    
    for note in sequence.notes:
        midi_note = note.pitch
        instrument = drum_mapping.get(midi_note, f"drum_{midi_note}")
        
        events.append({
            "time_seconds": float(note.start_time),
            "instrument": instrument,
            "velocity": int(note.velocity),
            "midi_note": midi_note,
            "confidence": 1.0,  # OaF doesn't provide confidence per-note
            "end_time": float(note.end_time),
        })
    
    # Sort by time
    events.sort(key=lambda e: e["time_seconds"])
    
    print(f"[OaF Drums] Detected {len(events)} drum hits")
    
    # Count by instrument
    inst_counts = {}
    for e in events:
        inst = e["instrument"]
        inst_counts[inst] = inst_counts.get(inst, 0) + 1
    print(f"[OaF Drums] Breakdown: {inst_counts}")
    
    return events
