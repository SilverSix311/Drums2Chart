# Drums2Chart

**ComfyUI nodes for automatic drum chart generation for YARG/Clone Hero**

Transform any audio/video into a playable drum chart using AI-powered transcription.

## 🎯 Goal

Input a song → Output a complete YARG-ready drum chart package

## 🔧 Pipeline

### Basic Pipeline (5 classes)
```
LoadAudio ─→ StemIsolate ─→ DrumTranscribe ─→ DrumMapping ─→ MIDIToChart ─→ PackageYARGChart
               (Demucs)        (ADTOF)                          (.chart)
```

### Enhanced Pipeline (7+ classes)
```
LoadAudio ─→ StemIsolate ─→ DrumTranscribe ─→ DrumRefine ─→ DrumMapping ─→ MIDIToChart
               (Demucs)        (ADTOF)         (7-class)
                  │                               ↑
                  └── crash/ride stems ───────────┘
```

The **DrumRefine** node expands ADTOF's 5 classes into 7+ classes:
- `hihat_closed` → `hihat_open` / `hihat_closed` (decay analysis)
- `crash` → `crash` / `ride` (stem loudness comparison)

### Model Loading Pattern

Just like SDXL checkpoints - load once, use many times:

```
LoadDrumModel ──► model ──┬──► DrumTranscribe (song 1)
                         ├──► DrumTranscribe (song 2)
                         └──► DrumTranscribe (song 3)
```

## 📦 Nodes

| Node | Description | Status |
|------|-------------|--------|
| `LoadDrumModel` | Load transcription model | ✅ Done |
| `UnloadDrumModel` | Free model from VRAM | ✅ Done |
| `StemIsolate` | Demucs stem separation (6 stems) | ✅ Done |
| `DrumTranscribe` | ADTOF AI drum transcription | ✅ Working |
| `DrumRefine` | Expand 5→7 classes (open/closed hh, crash/ride) | ✅ New |
| `DrumMapping` | Customize instrument → lane mapping | ✅ Done |
| `DrumMappingPreset` | Quick preset mappings | ✅ Done |
| `MIDIToChart` | Convert to .chart format | ✅ Done |
| `PackageYARGChart` | Bundle chart + stems + metadata | ✅ Done |
| `PreviewChart` | Visual preview of chart | ✅ Done |

### Dependencies

| Node | Source | Purpose |
|------|--------|---------|
| `LoadAudio` / `LoadVideo` | ComfyUI Core | Input |
| `AudioSeparateDemucs` | set-soft/AudioSeparation | Alternative stem separation |
| `AudioGetTempo` | christian-byrne/audio-separation-nodes | BPM detection |

## 🧠 AI Models

### ADTOF (Default)
- **Classes**: kick, snare, tom, hihat_closed, crash
- **Accuracy**: F1 0.85-0.94
- **Trained on**: Rhythm game data (Guitar Hero, Rock Band)
- **Framework**: PyTorch

### DrumRefine Enhancement
Using audio analysis (no extra ML models):
- **Hi-hat**: Decay curve analysis (slow decay = open)
- **Cymbal**: Stem loudness comparison + refractory periods
- **Based on**: [arxiv.org/html/2509.24853v1](https://arxiv.org/html/2509.24853v1)

## 🎮 YARG/Clone Hero Drum Mapping

| MIDI Note | Instrument | Chart Lane | Cymbal |
|-----------|------------|------------|--------|
| 36 (C1) | Kick | Orange (0) | No |
| 38 (D1) | Snare | Red (1) | No |
| 42 (F#1) | Closed Hi-Hat | Yellow (2) | Yes |
| 46 (A#1) | Open Hi-Hat | Yellow (2) | Yes |
| 47/45/43 | Toms | Blue/Green (3/4) | No |
| 49 | Crash | Green (4) | Yes |
| 51 | Ride | Blue (3) | Yes |
| 44 (G#1) | Hi-Hat Pedal | Orange 2nd (5) | No |

## 📁 Project Structure

```
Drums2Chart/
├── __init__.py              # ComfyUI node registration
├── nodes/
│   ├── load_drum_model.py   # Model loading/unloading
│   ├── stem_isolate.py      # Demucs separation
│   ├── drum_transcribe.py   # ADTOF transcription
│   ├── drum_refine.py       # 5→7 class expansion
│   ├── drum_mapping.py      # Lane assignment
│   ├── midi_to_chart.py     # .chart generation
│   ├── package_chart.py     # YARG packaging
│   └── preview_chart.py     # Visual preview
├── utils/
│   ├── adtof_integration.py # ADTOF model wrapper
│   ├── drum_refinement.py   # Hi-hat/cymbal analysis
│   └── oaf_drums_integration.py # (Future) OaF support
├── models/
│   └── drums2chart/         # Model weights
└── requirements.txt
```

## 🚀 Installation

```bash
# Clone to ComfyUI custom_nodes
cd ComfyUI/custom_nodes
git clone https://github.com/SilverSix311/Drums2Chart.git
cd Drums2Chart
pip install -r requirements.txt

# Download ADTOF weights
# Place adtof_frame_rnn.pth in ComfyUI/models/drums2chart/
```

## 💡 Tips

- **Better cymbal classification**: Use `htdemucs_6s` model for 6-stem separation, then connect crash/ride stems to DrumRefine
- **Polyrhythm songs**: Increase chart resolution to 384 or 480
- **Hi-hat accuracy**: Adjust `hihat_open_threshold` (default 0.70) - higher = stricter open detection
- **Manual cleanup**: Export to Moonscraper for final polish

## 📚 References

- [ADTOF](https://github.com/MZehren/ADTOF) — Drum transcription model
- [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch) — PyTorch port
- [7-class refinement paper](https://arxiv.org/html/2509.24853v1) — Hi-hat/cymbal heuristics
- [MIDI-CH](https://efhiii.github.io/midi-ch/) — MIDI to Clone Hero converter
- [audio-separation-nodes](https://github.com/christian-byrne/audio-separation-nodes-comfyui) — Demucs reference

## 📄 License

MIT
