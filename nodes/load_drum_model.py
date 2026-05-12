"""
LoadDrumModel Node

Loads drum transcription models for use with DrumTranscribe node.
Follows ComfyUI checkpoint loading pattern - model is loaded once and passed to inference nodes.
"""

import os
import torch
import folder_paths  # ComfyUI's folder management
from typing import Dict, Any, Tuple, Optional

# Register our models folder with ComfyUI
DRUMS2CHART_MODELS_DIR = os.path.join(folder_paths.models_dir, "drums2chart")
os.makedirs(DRUMS2CHART_MODELS_DIR, exist_ok=True)

# Add to ComfyUI's folder system so it can find our models
if "drums2chart" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["drums2chart"] = (
        [DRUMS2CHART_MODELS_DIR],
        {".pt", ".pth", ".onnx", ".safetensors", ".ckpt"}
    )


class LoadDrumModel:
    """
    Loads a drum transcription model from the models/drums2chart folder.
    
    Supported model types:
    - ADTOF (.pt/.pth) - Best accuracy for real music
    - Omnizart (.onnx) - Easy to use, good baseline
    - Onsets & Frames (.ckpt) - Google/Magenta baseline
    - Custom ONNX models
    
    Models are loaded once and cached, then passed to DrumTranscribe for inference.
    """
    
    MODEL_TYPES = ["auto", "adtof", "oaf_drums", "omnizart", "onsets_frames", "onnx_generic"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Get available model files
        model_files = cls._get_model_files()
        
        return {
            "required": {
                "model_name": (model_files, {
                    "tooltip": "Select model file from models/drums2chart folder"
                }),
            },
            "optional": {
                "model_type": (cls.MODEL_TYPES, {
                    "default": "auto",
                    "tooltip": "Model architecture type (auto-detect from filename if 'auto')"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load model on"
                }),
                "precision": (["fp32", "fp16", "bf16"], {
                    "default": "fp32",
                    "tooltip": "Model precision (fp16 uses less VRAM)"
                }),
            }
        }
    
    RETURN_TYPES = ("DRUM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = """
    Loads a drum transcription model for use with DrumTranscribe.
    
    Place model files in: ComfyUI/models/drums2chart/
    
    Supported formats:
    - .pt / .pth (PyTorch)
    - .onnx (ONNX Runtime)
    - .safetensors (SafeTensors)
    - .ckpt (Checkpoints)
    
    Model is loaded once and cached for efficient reuse.
    """
    
    # Class-level cache for loaded models
    _model_cache: Dict[str, Any] = {}
    
    @classmethod
    def _get_model_files(cls) -> list:
        """Get list of available model files"""
        try:
            files = folder_paths.get_filename_list("drums2chart")
            if not files:
                return ["[No models found - add to models/drums2chart/]"]
            return files
        except:
            return ["[No models found - add to models/drums2chart/]"]
    
    @classmethod
    def IS_CHANGED(cls, model_name: str, **kwargs) -> str:
        """Check if model file has changed (for caching)"""
        try:
            model_path = folder_paths.get_full_path("drums2chart", model_name)
            if model_path and os.path.exists(model_path):
                return str(os.path.getmtime(model_path))
        except:
            pass
        return ""

    def load_model(
        self,
        model_name: str,
        model_type: str = "auto",
        device: str = "auto",
        precision: str = "fp32",
    ) -> Tuple[Dict[str, Any]]:
        """
        Load drum transcription model.
        
        Returns:
            Tuple containing model dict with:
            - model: The loaded model object
            - model_type: Detected/specified model type
            - device: Device model is on
            - config: Model-specific configuration
        """
        # Check cache first
        cache_key = f"{model_name}_{device}_{precision}"
        if cache_key in self._model_cache:
            print(f"[Drums2Chart] Using cached model: {model_name}")
            return (self._model_cache[cache_key],)
        
        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get full path
        if model_name.startswith("["):
            raise ValueError("No model selected. Please add models to ComfyUI/models/drums2chart/")
        
        model_path = folder_paths.get_full_path("drums2chart", model_name)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_name}")
        
        print(f"[Drums2Chart] Loading model: {model_name} on {device}")
        
        # Auto-detect model type from filename
        if model_type == "auto":
            model_type = self._detect_model_type(model_name)
        
        # Load based on type
        if model_type == "adtof":
            model_obj, config = self._load_adtof(model_path, device, precision)
        elif model_type == "oaf_drums":
            model_obj, config = self._load_oaf_drums(model_path, device, precision)
        elif model_type == "omnizart":
            model_obj, config = self._load_omnizart(model_path, device, precision)
        elif model_type == "onsets_frames":
            model_obj, config = self._load_onsets_frames(model_path, device, precision)
        elif model_type == "onnx_generic":
            model_obj, config = self._load_onnx(model_path, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Build model container
        model_data = {
            "model": model_obj,
            "model_type": model_type,
            "model_name": model_name,
            "model_path": model_path,
            "device": device,
            "precision": precision,
            "config": config,
        }
        
        # Cache it
        self._model_cache[cache_key] = model_data
        
        print(f"[Drums2Chart] Model loaded: {model_type} ({precision}) on {device}")
        
        return (model_data,)
    
    def _detect_model_type(self, filename: str) -> str:
        """Detect model type from filename"""
        filename_lower = filename.lower()
        
        if "adtof" in filename_lower:
            return "adtof"
        elif "oaf" in filename_lower or "egmd" in filename_lower:
            return "oaf_drums"
        elif "omnizart" in filename_lower:
            return "omnizart"
        elif "onsets" in filename_lower or "frames" in filename_lower or "magenta" in filename_lower:
            return "onsets_frames"
        elif filename_lower.endswith(".onnx"):
            return "onnx_generic"
        else:
            # Default based on extension
            if filename_lower.endswith((".pt", ".pth")):
                return "adtof"  # Assume PyTorch = ADTOF
            elif filename_lower.endswith(".onnx"):
                return "onnx_generic"
            else:
                return "adtof"  # Default
    
    def _load_adtof(self, path: str, device: str, precision: str) -> Tuple[Any, Dict]:
        """Load ADTOF model"""
        from ..utils.adtof_integration import load_adtof_model
        
        try:
            result = load_adtof_model(weights_path=path, device=device)
            model_obj = result["model"]
            config = result["config"]
            
            # Apply precision
            if precision == "fp16" and device == "cuda":
                model_obj = model_obj.half()
            # bf16 requires specific GPU support
            # elif precision == "bf16":
            #     model_obj = model_obj.to(torch.bfloat16)
            
            return model_obj, config
            
        except ImportError as e:
            # ADTOF not installed - return placeholder with error
            print(f"[Drums2Chart] ADTOF not available: {e}")
            print("[Drums2Chart] Install with: pip install adtof-pytorch")
            
            config = {
                "sample_rate": 44100,
                "fps": 100,
                "instruments": ["kick", "snare", "hihat", "tom", "cymbal"],
                "_error": str(e),
            }
            
            model = {"_placeholder": True, "type": "adtof", "path": path, "_error": str(e)}
            
            return model, config
    
    def _load_oaf_drums(self, path: str, device: str, precision: str) -> Tuple[Any, Dict]:
        """Load Onsets & Frames Drums (E-GMD) model"""
        from ..utils.oaf_drums_integration import load_oaf_drums_model, check_magenta_available, EGMD_DRUM_MAPPING
        
        if not check_magenta_available():
            print("[Drums2Chart] Magenta not available")
            print("[Drums2Chart] Install with: pip install magenta")
            
            config = {
                "sample_rate": 16000,
                "instruments": list(EGMD_DRUM_MAPPING.values()),
                "_error": "Magenta not installed",
            }
            model = {"_placeholder": True, "type": "oaf_drums", "path": path, "_error": "Magenta not installed"}
            return model, config
        
        try:
            result = load_oaf_drums_model(
                checkpoint_path=path if path and not path.endswith(".pth") else None,
                model_dir=os.path.dirname(path) if path else DRUMS2CHART_MODELS_DIR,
                device=device,
            )
            
            config = {
                "sample_rate": result["sample_rate"],
                "instruments": list(result["drum_mapping"].values()),
                "drum_mapping": result["drum_mapping"],
                "hparams": result.get("hparams"),
            }
            
            return result, config
            
        except Exception as e:
            print(f"[Drums2Chart] OaF Drums load failed: {e}")
            config = {
                "sample_rate": 16000,
                "instruments": list(EGMD_DRUM_MAPPING.values()),
                "_error": str(e),
            }
            model = {"_placeholder": True, "type": "oaf_drums", "path": path, "_error": str(e)}
            return model, config
    
    def _load_omnizart(self, path: str, device: str, precision: str) -> Tuple[Any, Dict]:
        """Load Omnizart model"""
        # TODO: Implement Omnizart loading
        # omnizart uses its own model management
        
        config = {
            "sample_rate": 44100,
            "instruments": ["kick", "snare", "hihat"],
        }
        
        model = {"_placeholder": True, "type": "omnizart", "path": path}
        
        return model, config
    
    def _load_onsets_frames(self, path: str, device: str, precision: str) -> Tuple[Any, Dict]:
        """Load Google Magenta Onsets & Frames model"""
        # TODO: Implement O&F loading
        
        config = {
            "sample_rate": 16000,  # O&F uses 16kHz
            "hop_length": 512,
            "instruments": ["kick", "snare", "hihat", "tom"],
        }
        
        model = {"_placeholder": True, "type": "onsets_frames", "path": path}
        
        return model, config
    
    def _load_onnx(self, path: str, device: str) -> Tuple[Any, Dict]:
        """Load generic ONNX model"""
        try:
            import onnxruntime as ort
            
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(path, providers=providers)
            
            # Get model metadata
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            config = {
                "input_names": [i.name for i in inputs],
                "output_names": [o.name for o in outputs],
                "input_shapes": [i.shape for i in inputs],
            }
            
            return session, config
            
        except ImportError:
            raise ImportError("onnxruntime required for ONNX models: pip install onnxruntime-gpu")


class UnloadDrumModel:
    """
    Unloads a drum model from memory/cache.
    Useful for freeing VRAM when switching between models.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("DRUM_MODEL",),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "audio/Drums2Chart"
    OUTPUT_NODE = True
    
    def unload(self, model: Dict[str, Any]) -> Tuple:
        """Unload model from cache"""
        model_name = model.get("model_name", "unknown")
        
        # Clear from cache
        keys_to_remove = [k for k in LoadDrumModel._model_cache.keys() if model_name in k]
        for key in keys_to_remove:
            del LoadDrumModel._model_cache[key]
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[Drums2Chart] Unloaded model: {model_name}")
        
        return ()
