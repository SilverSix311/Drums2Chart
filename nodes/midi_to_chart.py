"""
MIDIToChart Node

Converts MIDI drum data to Clone Hero .chart format.
Handles drum lane mapping, time signature, and tempo track generation.
"""

from typing import Dict, Any, Tuple, List


class MIDIToChart:
    """
    Converts MIDI drum data to Clone Hero/YARG .chart format.
    
    Handles the complex mapping between:
    - General MIDI drum notes (36=kick, 38=snare, etc.)
    - Clone Hero lanes (Red, Yellow, Blue, Green, Orange + cymbals)
    
    Supports both 4-lane (Rock Band) and 5-lane modes.
    """
    
    LANE_MODES = ["5lane_prodrum", "4lane_rb"]
    DIFFICULTY_LEVELS = ["Expert", "Hard", "Medium", "Easy"]
    
    # Clone Hero note numbers for drums
    # Lane 0 = Kick (Orange pedal)
    # Lane 1 = Red (Snare)
    # Lane 2 = Yellow (Hi-hat/Cymbal)
    # Lane 3 = Blue (Tom/Cymbal)
    # Lane 4 = Green (Tom/Cymbal)
    # Lane 5 = Orange Cymbal (5-lane only)
    # +64 = Cymbal modifier
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "midi": ("MIDI_DATA",),
                "lane_mode": (cls.LANE_MODES, {"default": "5lane_prodrum"}),
                "resolution": ("INT", {
                    "default": 192,
                    "min": 96,
                    "max": 480,
                    "step": 48,
                    "tooltip": "Chart resolution (ticks per beat). 192 is standard."
                }),
            },
            "optional": {
                "tempo_bpm": ("FLOAT", {
                    "default": 0.0,  # 0 = auto-detect from MIDI
                    "min": 0.0,
                    "max": 300.0,
                    "step": 0.1,
                    "tooltip": "Override BPM (0 = use MIDI tempo)"
                }),
                "difficulty": (cls.DIFFICULTY_LEVELS, {"default": "Expert"}),
                "cymbal_heuristic": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-detect cymbals vs toms based on MIDI note"
                }),
            }
        }
    
    RETURN_TYPES = ("CHART_DATA", "STRING")
    RETURN_NAMES = ("chart", "chart_text")
    FUNCTION = "convert"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = """
    Converts MIDI drum data to Clone Hero .chart format.
    
    Lane mapping follows Pro Drums standard:
    - Kick → Orange (Pedal)
    - Snare → Red
    - Hi-Hat → Yellow Cymbal
    - Toms → Blue/Green
    - Crash/Ride → Blue/Green Cymbal
    """

    # GM Drum Map → Chart Lane mapping
    MIDI_TO_CHART = {
        # Kicks
        36: {"lane": 0, "cymbal": False},  # Kick
        35: {"lane": 0, "cymbal": False},  # Acoustic Bass Drum
        
        # Snare
        38: {"lane": 1, "cymbal": False},  # Snare
        40: {"lane": 1, "cymbal": False},  # Electric Snare
        37: {"lane": 1, "cymbal": False},  # Side Stick
        
        # Hi-Hat
        42: {"lane": 2, "cymbal": True},   # Closed Hi-Hat
        46: {"lane": 2, "cymbal": True},   # Open Hi-Hat
        44: {"lane": 0, "cymbal": False, "pedal2": True},  # Hi-Hat Pedal → 2nd pedal
        
        # Toms (Blue/Green)
        50: {"lane": 2, "cymbal": False},  # High Tom
        48: {"lane": 2, "cymbal": False},  # Hi-Mid Tom
        47: {"lane": 3, "cymbal": False},  # Low-Mid Tom
        45: {"lane": 3, "cymbal": False},  # Low Tom
        43: {"lane": 4, "cymbal": False},  # High Floor Tom
        41: {"lane": 4, "cymbal": False},  # Low Floor Tom
        
        # Cymbals
        49: {"lane": 3, "cymbal": True},   # Crash 1
        57: {"lane": 4, "cymbal": True},   # Crash 2
        51: {"lane": 4, "cymbal": True},   # Ride
        59: {"lane": 4, "cymbal": True},   # Ride 2
        53: {"lane": 4, "cymbal": True},   # Ride Bell
        55: {"lane": 3, "cymbal": True},   # Splash
        52: {"lane": 3, "cymbal": True},   # China
    }

    def convert(
        self,
        midi: Dict[str, Any],
        lane_mode: str = "5lane_prodrum",
        resolution: int = 192,
        tempo_bpm: float = 0.0,
        difficulty: str = "Expert",
        cymbal_heuristic: bool = True,
    ) -> Tuple[Dict, str]:
        """
        Convert MIDI to chart format.
        
        Returns:
            Tuple of (chart_data dict, chart_text string)
        """
        # Get tempo
        bpm = tempo_bpm if tempo_bpm > 0 else midi.get("tempo", 120)
        ticks_per_beat = midi.get("ticks_per_beat", 480)
        
        # Build chart sections
        song_section = self._build_song_section(bpm, resolution)
        sync_track = self._build_sync_track(bpm, resolution)
        
        # Convert MIDI events to chart notes
        drum_track = self._build_drum_track(
            midi["tracks"][0]["events"],
            bpm,
            ticks_per_beat,
            resolution,
            difficulty,
            cymbal_heuristic,
        )
        
        # Assemble chart text
        chart_text = self._assemble_chart(song_section, sync_track, drum_track, difficulty)
        
        chart_data = {
            "resolution": resolution,
            "bpm": bpm,
            "difficulty": difficulty,
            "note_count": len(midi["tracks"][0]["events"]),
            "text": chart_text,
        }
        
        return (chart_data, chart_text)
    
    def _build_song_section(self, bpm: float, resolution: int) -> str:
        """Build [Song] metadata section"""
        return f"""[Song]
{{
  Resolution = {resolution}
  Offset = 0
  Player2 = drums
  Difficulty = 0
  PreviewStart = 0
  PreviewEnd = 0
  Genre = "rock"
  MediaType = "cd"
  MusicStream = "song.ogg"
}}
"""

    def _build_sync_track(self, bpm: float, resolution: int) -> str:
        """Build [SyncTrack] tempo section"""
        # BPM is stored as microseconds per beat * 1000 in some formats
        # For .chart, it's just BPM * 1000 (milli-BPM)
        milli_bpm = int(bpm * 1000)
        return f"""[SyncTrack]
{{
  0 = TS 4
  0 = B {milli_bpm}
}}
"""

    def _build_drum_track(
        self,
        events: List[Dict],
        bpm: float,
        source_ticks_per_beat: int,
        target_resolution: int,
        difficulty: str,
        use_cymbal_heuristic: bool,
    ) -> str:
        """Convert MIDI events to chart drum notes"""
        lines = []
        
        seconds_per_beat = 60.0 / bpm
        
        for event in events:
            if event["type"] != "note_on":
                continue
                
            # Convert time to ticks
            time_seconds = event["time"]
            time_beats = time_seconds / seconds_per_beat
            tick = int(time_beats * target_resolution)
            
            # Check if event already has lane mapping (from DrumMapping node)
            if "chart_lane" in event or "is_cymbal" in event:
                # Use pre-computed mapping from DrumMapping
                lane = event.get("chart_lane", event.get("chart_lane_num", 1))
                is_cymbal = event.get("is_cymbal", False) if use_cymbal_heuristic else False
            else:
                # Fallback: Map MIDI note to chart lane
                midi_note = event["note"]
                mapping = self.MIDI_TO_CHART.get(midi_note)
                
                if mapping is None:
                    continue  # Unknown drum, skip
                
                lane = mapping["lane"]
                is_cymbal = mapping.get("cymbal", False) if use_cymbal_heuristic else False
            
            # Chart format: tick = N lane 0
            # For cymbals, lane number + 64 in some interpretations
            # Actually in .chart, cymbals are separate: tick = N lane 0 + tick = N lane+64 0
            
            note_value = lane
            lines.append(f"  {tick} = N {note_value} 0")
            
            # Add cymbal marker if applicable (yellow=2, blue=3, green=4)
            if is_cymbal and lane >= 2:
                lines.append(f"  {tick} = N {lane + 64} 0")
        
        # Sort by tick
        lines.sort(key=lambda x: int(x.strip().split(" = ")[0]))
        
        return "\n".join(lines)

    def _assemble_chart(
        self,
        song_section: str,
        sync_track: str,
        drum_track: str,
        difficulty: str,
    ) -> str:
        """Assemble complete .chart file"""
        section_name = f"[{difficulty}Drums]"
        
        return f"""{song_section}
{sync_track}
{section_name}
{{
{drum_track}
}}
"""
