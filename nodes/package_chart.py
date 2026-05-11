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
                "audio": ("AUDIO",),
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
                "album": ("STRING", {"default": "Unknown Album"}),
                "year": ("STRING", {"default": "2026"}),
                "genre": ("STRING", {"default": "Rock"}),
                "charter": ("STRING", {"default": "Drums2Chart AI"}),
                "drums_audio": ("AUDIO",),  # Isolated drums stem
                "album_art": ("IMAGE",),
                "preview_start_ms": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 600000,
                    "tooltip": "Preview start time in milliseconds"
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
        drums_audio: Dict[str, Any] = None,
        album_art=None,
        preview_start_ms: int = 0,
    ) -> Tuple[str, str]:
        """
        Package all components into song folder.
        
        Returns:
            Tuple of (output_folder_path, song_ini_content)
        """
        import torchaudio
        
        # Sanitize folder name
        safe_name = self._sanitize_filename(f"{artist} - {song_name}")
        folder_path = os.path.join(output_path, safe_name)
        
        # Create folder
        os.makedirs(folder_path, exist_ok=True)
        
        # Calculate song length from audio
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        song_length_ms = int((waveform.shape[-1] / sample_rate) * 1000)
        
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
        
        # Save audio as OGG
        audio_path = os.path.join(folder_path, "song.ogg")
        torchaudio.save(
            audio_path,
            audio["waveform"].squeeze(0),  # Remove batch dim
            audio["sample_rate"],
            format="ogg",
        )
        
        # Optional: Save drums stem
        if drums_audio is not None:
            drums_path = os.path.join(folder_path, "drums.ogg")
            torchaudio.save(
                drums_path,
                drums_audio["waveform"].squeeze(0),
                drums_audio["sample_rate"],
                format="ogg",
            )
        
        # Optional: Save album art
        if album_art is not None:
            # TODO: Convert tensor to PNG and save
            pass
        
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
