"""
Drums2Chart ComfyUI Nodes

Core nodes for the drum transcription → chart generation pipeline.
"""

from .drum_transcribe import DrumTranscribe
from .midi_to_chart import MIDIToChart
from .package_chart import PackageYARGChart

NODE_CLASS_MAPPINGS = {
    "DrumTranscribe": DrumTranscribe,
    "MIDIToChart": MIDIToChart,
    "PackageYARGChart": PackageYARGChart,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DrumTranscribe": "🥁 Drum Transcribe (Audio → MIDI)",
    "MIDIToChart": "📝 MIDI to Chart",
    "PackageYARGChart": "📦 Package YARG Chart",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
