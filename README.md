# Drums2Chart

**ComfyUI nodes for automatic drum chart generation for YARG/Clone Hero**

Transform any audio/video into a playable drum chart using AI-powered transcription.

## 🎯 Goal

Input a song → Output a complete YARG-ready drum chart package

## 🔧 Pipeline

```
┌───────────────┐
│ LoadDrumModel │─────────────────────────────────┐
└───────────────┘                                 │
                                                  ▼
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌─────────────┐    ┌──────────────┐
│ LoadAudio/  │───►│ AudioSepara- │───►│ DrumTranscribe │───►│ MIDIToChart │───►│ PackageChart │
│ LoadVideo   │    │ teDemucs     │    │                │    │             │    │              │
└─────────────┘    └──────────────┘    └────────────────┘    └─────────────┘    └──────────────┘
     Input              Stems              MIDI                 .chart            YARG Package
```

### Model Loading Pattern

Just like SDXL checkpoints - load once, use many times:

```
LoadDrumModel ──► model ──┬──► DrumTranscribe (song 1)
                         ├──► DrumTranscribe (song 2)
                         └──► DrumTranscribe (song 3)
```

## 📦 Nodes

### Drums2Chart Nodes

| Node | Description | Status |
|------|-------------|--------|
| `LoadDrumModel` | Load transcription model (like checkpoint loading) | ✅ Structure done |
| `UnloadDrumModel` | Free model from VRAM | ✅ Done |
| `DrumTranscribe` | AI drum transcription → MIDI | 🟡 Model integration pending |
| `MIDIToChart` | Convert MIDI to .chart format | ✅ Core logic done |
| `PackageYARGChart` | Bundle chart + audio + metadata | ✅ Core logic done |

### Compatible Nodes (Dependencies)

| Node | Source | Purpose |
|------|--------|---------|
| `LoadAudio` / `LoadVideo` | ComfyUI Core | Input |
| `AudioSeparateDemucs` | set-soft/AudioSeparation | Drum stem isolation |
| `AudioGetTempo` | christian-byrne/audio-separation-nodes | BPM detection |

## 🧠 AI Models

### Drum Transcription Options

| Model | Accuracy | Notes |
|-------|----------|-------|
| **ADTOF** | F1 0.85-0.94 | Best for real music, trained on rhythm game data |
| Onsets & Frames | F1 ~0.83 | Google/Magenta baseline |
| Omnizart | Good | Easiest CLI integration |

**Target: ADTOF** — best accuracy on non-synthetic drums

### Chart Format

Output follows Clone Hero `.chart` format:
- `[Song]` metadata (BPM, resolution, etc.)
- `[SyncTrack]` tempo changes
- `[ExpertDrums]` note data

## 📁 Project Structure

```
Drums2Chart/
├── __init__.py              # ComfyUI node registration
├── nodes/
│   ├── __init__.py
│   ├── drum_transcribe.py   # DrumTranscribe node
│   ├── midi_to_chart.py     # MIDIToChart node
│   └── package_chart.py     # PackageYARGChart node
├── models/
│   └── adtof/               # ADTOF model files
├── utils/
│   ├── chart_format.py      # .chart file generation
│   ├── midi_utils.py        # MIDI processing helpers
│   └── drum_mapping.py      # MIDI note → chart lane mapping
├── requirements.txt
└── README.md
```

## 🎮 YARG/Clone Hero Drum Mapping

| MIDI Note | Instrument | Chart Lane |
|-----------|------------|------------|
| 36 (C1) | Kick | Orange (Pedal 1) |
| 38 (D1) | Snare | Red |
| 42 (F#1) | Closed Hi-Hat | Yellow (Cymbal) |
| 46 (A#1) | Open Hi-Hat | Yellow (Cymbal) |
| 41/43/45 | Toms | Blue/Green |
| 49/51 | Crash/Ride | Blue/Green (Cymbal) |
| 44 (G#1) | Hi-Hat Pedal | Orange (Pedal 2) |

## 🚀 Installation

```bash
# Clone to ComfyUI custom_nodes
cd ComfyUI/custom_nodes
git clone https://github.com/SilverSix311/Drums2Chart.git
cd Drums2Chart
pip install -r requirements.txt
```

## 📚 References

- [ADTOF](https://github.com/MZehren/ADTOF) — Drum transcription model
- [MIDI-CH](https://efhiii.github.io/midi-ch/) — MIDI to Clone Hero converter (reference)
- [RyanOnTheInside](https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside) — Audio/MIDI node patterns
- [audio-separation-nodes](https://github.com/christian-byrne/audio-separation-nodes-comfyui) — Demucs integration reference

## 📄 License

MIT
