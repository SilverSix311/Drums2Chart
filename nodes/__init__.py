"""
Drums2Chart ComfyUI Nodes

Core nodes for the drum transcription → chart generation pipeline.
"""

from .load_drum_model import LoadDrumModel, UnloadDrumModel
from .stem_isolate import StemIsolate, StemIsolateSimple
from .drum_transcribe import DrumTranscribe
from .drum_mapping import DrumMapping, DrumMappingPreset
from .drum_refine import DrumRefine
from .preview_chart import PreviewChart, PreviewChartAnimated
from .midi_to_chart import MIDIToChart
from .package_chart import PackageYARGChart

NODE_CLASS_MAPPINGS = {
    # Model management
    "LoadDrumModel": LoadDrumModel,
    "UnloadDrumModel": UnloadDrumModel,
    # Stem separation
    "StemIsolate": StemIsolate,
    "StemIsolateSimple": StemIsolateSimple,
    # Transcription
    "DrumTranscribe": DrumTranscribe,
    # Refinement (5-class → 7-class expansion)
    "DrumRefine": DrumRefine,
    # Mapping (customize note → lane assignment)
    "DrumMapping": DrumMapping,
    "DrumMappingPreset": DrumMappingPreset,
    # Preview
    "PreviewChart": PreviewChart,
    "PreviewChartAnimated": PreviewChartAnimated,
    # Charting & packaging
    "MIDIToChart": MIDIToChart,
    "PackageYARGChart": PackageYARGChart,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDrumModel": "🎚️ Load Drum Model",
    "UnloadDrumModel": "🗑️ Unload Drum Model",
    "StemIsolate": "🎛️ Stem Isolate (Demucs)",
    "StemIsolateSimple": "🥁 Stem Isolate (Simple)",
    "DrumTranscribe": "🥁 Drum Transcribe",
    "DrumRefine": "✨ Drum Refine (7-class)",
    "DrumMapping": "🎯 Drum Mapping",
    "DrumMappingPreset": "🎯 Drum Mapping (Preset)",
    "PreviewChart": "👁️ Preview Chart",
    "PreviewChartAnimated": "🎬 Preview Chart (Animated)",
    "MIDIToChart": "📝 MIDI → Chart",
    "PackageYARGChart": "📦 Package YARG Chart",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
