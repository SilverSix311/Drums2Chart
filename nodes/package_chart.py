"""
PackageYARGChart Node

Bundles chart, audio, and metadata into a complete YARG-ready song folder.
"""

import os
from typing import Dict, Any, Tuple


class PackageYARGChart:
    """
    Packages all components into a YARG/Clone Hero song folder.
    
    Creates the standard folder structure:
    ```
    SongName/
    ├── song.ini       # Metadata
    ├── notes.chart    # The chart file
    ├── song.ogg       # Full mix audio
    ├── drums.ogg      # (optional) Isolated drums
    └── album.png      # (optional) Album art
    ```
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "chart": ("CHART_DATA",),
                "audio": ("AUDIO",),  # Full mix (song.ogg)
                "song_name": ("STRING", {
                    "default": "Unknown Song",
                    "tooltip": "Song title"
                }),
                "artist": ("STRING", {
                    "default": "Unknown Artist",
                    "tooltip": "Artist name"
                }),
                "output_path": ("STRING", {
                    "default": "./output/charts",
                    "tooltip": "Base folder for chart output"
                }),
            },
            "optional": {
                # Metadata
                "album": ("STRING", {"default": "Unknown Album"}),
                "year": ("STRING", {"default": "2026"}),
                "genre": ("STRING", {"default": "Rock"}),
                "charter": ("STRING", {"default": "Drums2Chart AI"}),
                
                # Stems - all optional, will be saved if provided
                "drums_stem": ("AUDIO",),      # drums.ogg
                "bass_stem": ("AUDIO",),       # bass.ogg
                "guitar_stem": ("AUDIO",),     # guitar.ogg (from "other" stem)
                "vocals_stem": ("AUDIO",),     # vocals.ogg
                "keys_stem": ("AUDIO",),       # keys.ogg
                "backing_stem": ("AUDIO",),    # backing.ogg (everything minus drums)
                
                # Additional options
                "album_art": ("IMAGE",),
                "preview_start_ms": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 600000,
                    "tooltip": "Preview start time in milliseconds"
                }),
                "include_stems": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include stem files in output (if provided)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_folder", "song_ini")
    FUNCTION = "package"
    CATEGORY = "audio/Drums2Chart"
    OUTPUT_NODE = True
    
    DESCRIPTION = """
    Packages chart + audio into a YARG-ready song folder.
    
    Creates:
    - song.ini (metadata)
    - notes.chart (the chart)
    - song.ogg (audio)
    - Optional: drums.ogg, album.png
    """

    def package(
        self,
        chart: Dict[str, Any],
        audio: Dict[str, Any],
        song_name: str,
        artist: str,
        output_path: str,
        album: str = "Unknown Album",
        year: str = "2026",
        genre: str = "Rock",
        charter: str = "Drums2Chart AI",
        drums_stem: Dict[str, Any] = None,
        bass_stem: Dict[str, Any] = None,
        guitar_stem: Dict[str, Any] = None,
        vocals_stem: Dict[str, Any] = None,
        keys_stem: Dict[str, Any] = None,
        backing_stem: Dict[str, Any] = None,
        album_art=None,
        preview_start_ms: int = 0,
        include_stems: bool = True,
    ) -> Tuple[str, str]:
        """
        Package all components into song folder.
        
        Returns:
            Tuple of (output_folder_path, song_ini_content)
        """
        import scipy.io.wavfile as wavfile
        import numpy as np
        import subprocess
        
        # Sanitize folder name
        safe_name = self._sanitize_filename(f"{artist} - {song_name}")
        folder_path = os.path.join(output_path, safe_name)
        
        # Create folder
        os.makedirs(folder_path, exist_ok=True)
        
        # Calculate song length from audio
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        song_length_ms = int((waveform.shape[-1] / sample_rate) * 1000)
        
        # Track which stems are included
        stems_included = []
        
        # Helper to save audio
        def save_audio_ogg(filepath: str, audio_dict: dict):
            """Save audio dict to OGG via WAV + ffmpeg"""
            wav = audio_dict["waveform"].squeeze()  # Remove batch dim
            sr = audio_dict["sample_rate"]
            
            # Convert to numpy
            if hasattr(wav, 'cpu'):
                wav = wav.cpu().numpy()
            
            # Ensure shape is [channels, samples] then transpose for scipy
            if wav.ndim == 1:
                wav = wav.reshape(1, -1)  # [1, samples]
            elif wav.shape[0] > wav.shape[1]:  # [samples, channels]
                wav = wav.T
            
            # Convert to int16 for WAV
            wav = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav * 32767).astype(np.int16)
            
            # Transpose to [samples, channels] for scipy
            wav_int16 = wav_int16.T
            
            # Save as WAV first
            wav_path = filepath.replace('.ogg', '.wav')
            wavfile.write(wav_path, sr, wav_int16)
            
            # Convert to OGG using ffmpeg if available
            try:
                result = subprocess.run(
                    ['ffmpeg', '-y', '-i', wav_path, '-c:a', 'libvorbis', '-q:a', '6', filepath],
                    capture_output=True, timeout=120
                )
                if result.returncode == 0:
                    os.remove(wav_path)  # Clean up WAV
                else:
                    # ffmpeg failed, rename WAV to final path
                    print(f"[Drums2Chart] ffmpeg failed, keeping WAV: {wav_path}")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                # ffmpeg not available, just keep WAV
                print(f"[Drums2Chart] ffmpeg not found, saved as WAV: {wav_path}")
        
        # Build song.ini
        song_ini = self._build_song_ini(
            song_name=song_name,
            artist=artist,
            album=album,
            year=year,
            genre=genre,
            charter=charter,
            song_length=song_length_ms,
            preview_start=preview_start_ms,
            difficulty=chart.get("difficulty", "Expert"),
        )
        
        # Write song.ini
        ini_path = os.path.join(folder_path, "song.ini")
        with open(ini_path, "w", encoding="utf-8") as f:
            f.write(song_ini)
        
        # Write notes.chart
        chart_path = os.path.join(folder_path, "notes.chart")
        with open(chart_path, "w", encoding="utf-8") as f:
            f.write(chart["text"])
        
        # Save main audio as OGG
        audio_path = os.path.join(folder_path, "song.ogg")
        save_audio_ogg(audio_path, audio)
        
        # Save stems if provided and requested
        if include_stems:
            stems = {
                "drums.ogg": drums_stem,
                "bass.ogg": bass_stem,
                "guitar.ogg": guitar_stem,
                "vocals.ogg": vocals_stem,
                "keys.ogg": keys_stem,
                "backing.ogg": backing_stem,
            }
            
            for filename, stem_audio in stems.items():
                if stem_audio is not None:
                    stem_path = os.path.join(folder_path, filename)
                    save_audio_ogg(stem_path, stem_audio)
                    stems_included.append(filename)
                    print(f"[Drums2Chart] Saved stem: {filename}")
        
        # Save album art
        if album_art is not None:
            try:
                from PIL import Image
                import numpy as np
                
                # Convert tensor to PIL Image
                if hasattr(album_art, 'cpu'):
                    img_np = album_art.cpu().numpy()
                else:
                    img_np = np.array(album_art)
                
                # Handle different formats
                if img_np.ndim == 4:
                    img_np = img_np[0]  # Remove batch
                if img_np.shape[0] in [1, 3, 4]:  # CHW format
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                # Scale to 0-255 if needed
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                
                img = Image.fromarray(img_np)
                art_path = os.path.join(folder_path, "album.png")
                img.save(art_path)
                print(f"[Drums2Chart] Saved album art")
            except Exception as e:
                print(f"[Drums2Chart] Could not save album art: {e}")
        
        print(f"[Drums2Chart] Chart packaged to: {folder_path}")
        if stems_included:
            print(f"[Drums2Chart] Stems: {', '.join(stems_included)}")
        
        return (folder_path, song_ini)
    
    def _sanitize_filename(self, name: str) -> str:
        """Remove/replace characters invalid in filenames"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, "_")
        return name.strip()
    
    def _build_song_ini(
        self,
        song_name: str,
        artist: str,
        album: str,
        year: str,
        genre: str,
        charter: str,
        song_length: int,
        preview_start: int,
        difficulty: str,
    ) -> str:
        """Build song.ini metadata file"""
        # Difficulty rating (1-6 scale, we default to 5 for Expert)
        diff_rating = {
            "Expert": 5,
            "Hard": 4,
            "Medium": 3,
            "Easy": 2,
        }.get(difficulty, 5)
        
        return f"""[song]
name = {song_name}
artist = {artist}
album = {album}
genre = {genre}
year = {year}
charter = {charter}
song_length = {song_length}
preview_start_time = {preview_start}
diff_drums = {diff_rating}
pro_drums = True
icon = drums2chart
"""
