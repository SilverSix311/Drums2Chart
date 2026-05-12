"""
DrumMapping Node

Configures how detected drum instruments map to chart lanes/colors.
Goes between DrumTranscribe and MIDIToChart for customization.
"""

from typing import Dict, Any, Tuple, List


class DrumMapping:
    """
    Configure drum-to-lane mapping for chart generation.
    
    ADTOF detects: kick, snare, hihat_closed, tom, crash
    
    Chart lanes:
    - Red (1): Typically snare
    - Yellow (2): Typically hi-hat (cymbal)
    - Blue (3): Tom or cymbal
    - Green (4): Tom or cymbal  
    - Orange (0): Kick pedal
    - 2nd Pedal: Hi-hat pedal (optional)
    
    Each instrument can be mapped to a lane + cymbal flag.
    """
    
    LANES = ["red", "yellow", "blue", "green", "orange_kick", "orange_hihat_pedal"]
    INSTRUMENTS = ["kick", "snare", "hihat_closed", "hihat_open", "tom_high", "tom_mid", "tom_low", "tom_floor", "crash", "ride", "china", "splash"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "events": ("DRUM_EVENTS",),  # From DrumTranscribe
            },
            "optional": {
                # Core mappings
                "kick_lane": (["orange_kick"], {"default": "orange_kick"}),
                "snare_lane": (["red", "yellow", "blue", "green"], {"default": "red"}),
                "hihat_lane": (["yellow", "blue", "green", "red"], {"default": "yellow"}),
                "hihat_is_cymbal": ("BOOLEAN", {"default": True}),
                
                # Tom mappings
                "tom_high_lane": (["yellow", "blue", "green", "red"], {"default": "blue"}),
                "tom_mid_lane": (["yellow", "blue", "green", "red"], {"default": "blue"}),
                "tom_low_lane": (["yellow", "blue", "green", "red"], {"default": "green"}),
                "tom_floor_lane": (["yellow", "blue", "green", "red"], {"default": "green"}),
                "toms_are_cymbals": ("BOOLEAN", {"default": False}),
                
                # Cymbal mappings
                "crash_lane": (["yellow", "blue", "green", "red"], {"default": "green"}),
                "crash_is_cymbal": ("BOOLEAN", {"default": True}),
                "ride_lane": (["yellow", "blue", "green", "red"], {"default": "blue"}),
                "ride_is_cymbal": ("BOOLEAN", {"default": True}),
                
                # Velocity handling
                "accent_threshold": ("INT", {
                    "default": 110,
                    "min": 1,
                    "max": 127,
                    "tooltip": "Velocity above this = accent note"
                }),
                "ghost_threshold": ("INT", {
                    "default": 40,
                    "min": 1,
                    "max": 127,
                    "tooltip": "Velocity below this = ghost note (may be filtered)"
                }),
                "filter_ghosts": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove ghost notes from chart"
                }),
            }
        }
    
    RETURN_TYPES = ("MIDI_DATA", "DRUM_EVENTS")
    RETURN_NAMES = ("midi", "mapped_events")
    FUNCTION = "apply_mapping"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = """
    Customize how detected drums map to chart lanes.
    
    Use this between DrumTranscribe and MIDIToChart to:
    - Reassign instruments to different colors
    - Control cymbal vs tom designation
    - Filter ghost notes
    - Set accent thresholds
    
    Pro tip: For complex songs, you might want high toms on blue
    and low toms on green for playability.
    """
    
    # Lane to MIDI note mapping for chart
    LANE_TO_MIDI = {
        "red": 1,
        "yellow": 2,
        "blue": 3,
        "green": 4,
        "orange_kick": 0,
        "orange_hihat_pedal": 5,  # 2nd pedal
    }

    def apply_mapping(
        self,
        events: List[Dict],
        kick_lane: str = "orange_kick",
        snare_lane: str = "red",
        hihat_lane: str = "yellow",
        hihat_is_cymbal: bool = True,
        tom_high_lane: str = "blue",
        tom_mid_lane: str = "blue",
        tom_low_lane: str = "green",
        tom_floor_lane: str = "green",
        toms_are_cymbals: bool = False,
        crash_lane: str = "green",
        crash_is_cymbal: bool = True,
        ride_lane: str = "blue",
        ride_is_cymbal: bool = True,
        accent_threshold: int = 110,
        ghost_threshold: int = 40,
        filter_ghosts: bool = False,
    ) -> Tuple[Dict, List]:
        """Apply custom mapping to drum events"""
        
        # Build mapping table (covers both ADTOF and OaF Drums outputs)
        mapping = {
            # Kick
            "kick": {"lane": kick_lane, "cymbal": False},
            
            # Snare variants
            "snare": {"lane": snare_lane, "cymbal": False},
            "snare_rimshot": {"lane": snare_lane, "cymbal": False},
            "snare_xstick": {"lane": snare_lane, "cymbal": False},
            
            # Hi-hat variants
            "hihat_closed": {"lane": hihat_lane, "cymbal": hihat_is_cymbal},
            "hihat_open": {"lane": hihat_lane, "cymbal": hihat_is_cymbal},
            "hihat_pedal": {"lane": "orange_hihat_pedal", "cymbal": False},
            
            # Tom variants (OaF gives us individual toms!)
            "tom": {"lane": tom_mid_lane, "cymbal": toms_are_cymbals},  # Generic tom
            "tom_high": {"lane": tom_high_lane, "cymbal": toms_are_cymbals},
            "tom_mid": {"lane": tom_mid_lane, "cymbal": toms_are_cymbals},
            "tom_low": {"lane": tom_low_lane, "cymbal": toms_are_cymbals},
            "tom_floor": {"lane": tom_floor_lane, "cymbal": toms_are_cymbals},
            
            # Cymbal variants
            "crash": {"lane": crash_lane, "cymbal": crash_is_cymbal},
            "crash_2": {"lane": crash_lane, "cymbal": crash_is_cymbal},
            "ride": {"lane": ride_lane, "cymbal": ride_is_cymbal},
            "ride_2": {"lane": ride_lane, "cymbal": ride_is_cymbal},
            "ride_bell": {"lane": ride_lane, "cymbal": ride_is_cymbal},
            "china": {"lane": crash_lane, "cymbal": True},
            "splash": {"lane": crash_lane, "cymbal": True},
        }
        
        mapped_events = []
        
        for event in events:
            velocity = event.get("velocity", 80)
            
            # Filter ghost notes if requested
            if filter_ghosts and velocity < ghost_threshold:
                continue
            
            instrument = event.get("instrument", "unknown")
            
            # Get mapping (default to snare if unknown)
            inst_mapping = mapping.get(instrument, {"lane": "red", "cymbal": False})
            
            # Create mapped event
            mapped_event = {
                **event,
                "chart_lane": inst_mapping["lane"],
                "chart_lane_num": self.LANE_TO_MIDI[inst_mapping["lane"]],
                "is_cymbal": inst_mapping["cymbal"],
                "is_accent": velocity >= accent_threshold,
                "is_ghost": velocity < ghost_threshold,
            }
            
            mapped_events.append(mapped_event)
        
        # Build MIDI data structure for MIDIToChart
        midi_data = self._events_to_midi(mapped_events)
        
        print(f"[Drums2Chart] Mapped {len(mapped_events)} events")
        
        return (midi_data, mapped_events)
    
    def _events_to_midi(self, events: List[Dict]) -> Dict:
        """Convert mapped events to MIDI structure"""
        midi_data = {
            "ticks_per_beat": 480,
            "tempo": 120,  # Will be overridden if tempo is known
            "tracks": [{
                "name": "Drums",
                "channel": 9,
                "events": []
            }]
        }
        
        for event in events:
            midi_data["tracks"][0]["events"].append({
                "type": "note_on",
                "time": event["time_seconds"],
                "note": event.get("midi_note", 36),
                "velocity": event.get("velocity", 80),
                "duration": 0.05,
                "chart_lane": event.get("chart_lane_num", 1),
                "is_cymbal": event.get("is_cymbal", False),
                "is_accent": event.get("is_accent", False),
            })
        
        return midi_data


class DrumMappingPreset:
    """
    Quick preset mappings for common setups.
    
    Use this for one-click standard configurations.
    """
    
    PRESETS = [
        "rock_band_pro",      # Standard RB pro drums
        "clone_hero_4lane",   # CH 4-lane
        "metal_double_bass",  # Kick-heavy, toms spread
        "jazz_ride_focus",    # Ride on yellow, crashes on green
    ]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "events": ("DRUM_EVENTS",),
                "preset": (cls.PRESETS, {"default": "rock_band_pro"}),
            }
        }
    
    RETURN_TYPES = ("MIDI_DATA", "DRUM_EVENTS")
    RETURN_NAMES = ("midi", "mapped_events")
    FUNCTION = "apply_preset"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = "Apply a preset drum mapping configuration."

    def apply_preset(
        self,
        events: List[Dict],
        preset: str = "rock_band_pro",
    ) -> Tuple[Dict, List]:
        """Apply preset mapping"""
        
        # Preset configurations
        presets = {
            "rock_band_pro": {
                "snare_lane": "red",
                "hihat_lane": "yellow",
                "tom_high_lane": "yellow",
                "tom_mid_lane": "blue",
                "tom_low_lane": "green",
                "crash_lane": "green",
                "ride_lane": "blue",
            },
            "clone_hero_4lane": {
                "snare_lane": "red",
                "hihat_lane": "yellow",
                "tom_high_lane": "blue",
                "tom_mid_lane": "blue",
                "tom_low_lane": "green",
                "crash_lane": "green",
                "ride_lane": "yellow",
            },
            "metal_double_bass": {
                "snare_lane": "red",
                "hihat_lane": "yellow",
                "tom_high_lane": "blue",
                "tom_mid_lane": "blue",
                "tom_low_lane": "green",
                "tom_floor_lane": "green",
                "crash_lane": "green",
                "ride_lane": "blue",
            },
            "jazz_ride_focus": {
                "snare_lane": "red",
                "hihat_lane": "yellow",
                "tom_high_lane": "blue",
                "tom_mid_lane": "green",
                "tom_low_lane": "green",
                "crash_lane": "green",
                "ride_lane": "yellow",
            },
        }
        
        config = presets.get(preset, presets["rock_band_pro"])
        
        # Use the full mapping node
        mapper = DrumMapping()
        return mapper.apply_mapping(
            events=events,
            hihat_is_cymbal=True,
            toms_are_cymbals=False,
            crash_is_cymbal=True,
            ride_is_cymbal=True,
            accent_threshold=110,
            ghost_threshold=40,
            filter_ghosts=False,
            **config,
        )
