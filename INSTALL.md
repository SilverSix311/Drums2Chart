# Installation Guide

## Quick Install

```bash
# 1. Clone to ComfyUI custom_nodes
cd ComfyUI/custom_nodes
git clone https://github.com/SilverSix311/Drums2Chart.git

# 2. Install dependencies
cd Drums2Chart
pip install -r requirements.txt

# 3. Restart ComfyUI
```

The ADTOF model weights are included with the `adtof-pytorch` package — no manual download needed!

## Dependencies

### Required
- **ComfyUI** (obviously)
- **PyTorch** ≥2.0.0
- **adtof-pytorch** — Drum transcription model (installs with requirements.txt)
- **librosa** — Audio processing
- **pretty_midi** — MIDI output

### Optional (for better results)
- **AudioSeparation** nodes by set-soft — For stem separation with Demucs
  - Install via ComfyUI Manager: search "AudioSeparation"
  - Or: https://github.com/set-soft/comfyui-audio-separation

## Model Files

### ADTOF (Included)
The ADTOF Frame_RNN model weights are bundled with the `adtof-pytorch` package. No manual download required.

If you want to use custom trained weights:
1. Place `.pth` files in `ComfyUI/models/drums2chart/`
2. Select them in the LoadDrumModel node

### Future Models
When we add support for Onsets & Frames or other models, their weights will go in the same folder:
```
ComfyUI/models/drums2chart/
├── adtof_custom_trained.pth
├── onsets_frames_checkpoint.ckpt
└── my_model.onnx
```

## Troubleshooting

### "ADTOF not available" error
```bash
pip install adtof-pytorch
```

### Import errors after install
Restart ComfyUI completely (not just reload custom nodes).

### CUDA out of memory
In LoadDrumModel, set:
- **device:** cpu
- Or **precision:** fp16

### Model dropdown shows "[No models found...]"
This is normal if you haven't added custom models. The default ADTOF weights are embedded in the package and don't need to be in the models folder.

To use the dropdown, add custom model files to `ComfyUI/models/drums2chart/`

## Verify Installation

After restarting ComfyUI:
1. Right-click in the workflow canvas
2. Search for "Drums2Chart" or "🥁"
3. You should see:
   - 🎚️ Load Drum Model
   - 🥁 Drum Transcribe
   - 📝 MIDI → Chart
   - 📦 Package YARG Chart

## Next Steps

Check out `examples/` folder for example workflows (coming soon).

For workflow setup, see the main README.md
