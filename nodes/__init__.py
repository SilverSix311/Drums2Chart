"""
Drums2Chart ComfyUI Nodes

Core nodes for the drum transcription → chart generation pipeline.
"""

from .load_drum_model import LoadDrumModel, UnloadDrumModel
from .drum_isolate import DrumIsolate, DrumIsolateSimple
from .drum_transcribe import DrumTranscribe
from .midi_to_chart import MIDIToChart
from .package_chart import PackageYARGChart

NODE_CLASS_MAPPINGS = {
    # Model management
    "LoadDrumModel": LoadDrumModel,
    "UnloadDrumModel": UnloadDrumModel,
    # Isolation
    "DrumIsolate": DrumIsolate,
    "DrumIsolateSimple": DrumIsolateSimple,
    # Transcription & charting
    "DrumTranscribe": DrumTranscribe,
    "MIDIToChart": MIDIToChart,
    "PackageYARGChart": PackageYARGChart,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDrumModel": "🎚️ Load Drum Model",
    "UnloadDrumModel": "🗑️ Unload Drum Model",
    "DrumIsolate": "🎛️ Drum Isolate (Demucs)",
    "DrumIsolateSimple": "🥁 Drum Isolate (Simple)",
    "DrumTranscribe": "🥁 Drum Transcribe",
    "MIDIToChart": "📝 MIDI → Chart",
    "PackageYARGChart": "📦 Package YARG Chart",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
