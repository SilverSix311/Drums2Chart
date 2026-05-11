"""
PreviewChart Node

Visualizes drum chart notes before export.
Great for checking mapping and timing before committing to a full chart.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, List
from PIL import Image, ImageDraw, ImageFont


class PreviewChart:
    """
    Generates a visual preview of the drum chart.
    
    Shows notes on a timeline with color-coded lanes,
    cymbal indicators, and accent/ghost styling.
    
    Output is an IMAGE that can be viewed in ComfyUI.
    """
    
    # Lane colors matching Clone Hero/Rock Band
    LANE_COLORS = {
        0: (255, 165, 0),    # Orange (kick)
        1: (255, 50, 50),    # Red (snare)
        2: (255, 255, 50),   # Yellow (hihat)
        3: (50, 150, 255),   # Blue (tom/cymbal)
        4: (50, 255, 50),    # Green (tom/cymbal)
        5: (255, 165, 0),    # Orange 2nd pedal
    }
    
    LANE_NAMES = {
        0: "Kick",
        1: "Snare", 
        2: "Yellow",
        3: "Blue",
        4: "Green",
        5: "HH Pedal",
    }
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "events": ("DRUM_EVENTS",),
            },
            "optional": {
                "width": ("INT", {
                    "default": 1920,
                    "min": 640,
                    "max": 4096,
                    "tooltip": "Image width in pixels"
                }),
                "height": ("INT", {
                    "default": 400,
                    "min": 200,
                    "max": 1080,
                    "tooltip": "Image height in pixels"
                }),
                "seconds_per_screen": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 1.0,
                    "tooltip": "How many seconds of audio to show per screen width"
                }),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.5,
                    "tooltip": "Start time in seconds"
                }),
                "show_cymbal_rings": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show rings around cymbal notes"
                }),
                "show_accents": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show accent notes with larger size"
                }),
                "show_ghosts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show ghost notes with transparency"
                }),
                "dark_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Dark background (like Clone Hero)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "stats")
    FUNCTION = "generate_preview"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = """
    Visualize drum notes before chart export.
    
    Shows:
    - Color-coded lanes (kick=orange, snare=red, etc.)
    - Cymbal notes with rings
    - Accent notes larger
    - Ghost notes transparent
    
    Use to verify mapping looks correct before packaging.
    """

    def generate_preview(
        self,
        events: List[Dict],
        width: int = 1920,
        height: int = 400,
        seconds_per_screen: float = 10.0,
        start_time: float = 0.0,
        show_cymbal_rings: bool = True,
        show_accents: bool = True,
        show_ghosts: bool = True,
        dark_mode: bool = True,
    ) -> Tuple[torch.Tensor, str]:
        """Generate chart preview image"""
        
        # Background color
        bg_color = (30, 30, 35) if dark_mode else (240, 240, 245)
        line_color = (60, 60, 70) if dark_mode else (200, 200, 210)
        text_color = (200, 200, 200) if dark_mode else (60, 60, 60)
        
        # Create image
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Layout
        margin_left = 80
        margin_right = 20
        margin_top = 40
        margin_bottom = 30
        
        chart_width = width - margin_left - margin_right
        chart_height = height - margin_top - margin_bottom
        
        # Calculate time range
        end_time = start_time + seconds_per_screen
        
        # Filter events in visible range
        visible_events = [
            e for e in events 
            if start_time <= e.get("time_seconds", 0) <= end_time
        ]
        
        # Draw lanes
        num_lanes = 5
        lane_height = chart_height / num_lanes
        
        for i in range(num_lanes):
            y = margin_top + i * lane_height
            
            # Lane background stripe
            if i % 2 == 0:
                stripe_color = (40, 40, 45) if dark_mode else (230, 230, 235)
                draw.rectangle(
                    [margin_left, y, width - margin_right, y + lane_height],
                    fill=stripe_color
                )
            
            # Lane label
            lane_idx = [0, 1, 2, 3, 4][i]  # Kick at top, then snare, yellow, blue, green
            lane_name = self.LANE_NAMES.get(lane_idx, f"Lane {lane_idx}")
            lane_color = self.LANE_COLORS.get(lane_idx, (150, 150, 150))
            
            # Draw colored indicator
            draw.rectangle(
                [5, y + lane_height/2 - 8, 25, y + lane_height/2 + 8],
                fill=lane_color
            )
            
            # Draw label
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            draw.text((30, y + lane_height/2 - 6), lane_name, fill=text_color, font=font)
        
        # Draw time markers
        time_step = 1.0  # Every second
        t = start_time
        while t <= end_time:
            x = margin_left + ((t - start_time) / seconds_per_screen) * chart_width
            
            # Vertical line
            draw.line([(x, margin_top), (x, height - margin_bottom)], fill=line_color, width=1)
            
            # Time label
            draw.text((x - 10, height - margin_bottom + 5), f"{t:.1f}s", fill=text_color, font=font)
            
            t += time_step
        
        # Draw notes
        note_radius_normal = 8
        note_radius_accent = 12
        note_radius_ghost = 5
        
        # Map lane numbers to Y positions
        lane_to_row = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0}  # 5 (hihat pedal) shares with kick
        
        for event in visible_events:
            time_sec = event.get("time_seconds", 0)
            lane = event.get("chart_lane_num", 1)
            is_cymbal = event.get("is_cymbal", False)
            is_accent = event.get("is_accent", False)
            is_ghost = event.get("is_ghost", False)
            velocity = event.get("velocity", 80)
            
            # Calculate position
            x = margin_left + ((time_sec - start_time) / seconds_per_screen) * chart_width
            row = lane_to_row.get(lane, 1)
            y = margin_top + row * lane_height + lane_height / 2
            
            # Get color
            color = self.LANE_COLORS.get(lane, (150, 150, 150))
            
            # Determine size
            if is_accent and show_accents:
                radius = note_radius_accent
            elif is_ghost and show_ghosts:
                radius = note_radius_ghost
            else:
                radius = note_radius_normal
            
            # Apply transparency for ghosts (darken color)
            if is_ghost and show_ghosts:
                color = tuple(int(c * 0.5) for c in color)
            
            # Draw note
            bbox = [x - radius, y - radius, x + radius, y + radius]
            
            if is_cymbal:
                # Diamond shape for cymbals
                points = [
                    (x, y - radius),      # top
                    (x + radius, y),      # right
                    (x, y + radius),      # bottom
                    (x - radius, y),      # left
                ]
                draw.polygon(points, fill=color)
                
                # Cymbal ring
                if show_cymbal_rings:
                    draw.ellipse(
                        [x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3],
                        outline=color,
                        width=2
                    )
            else:
                # Circle for drums
                draw.ellipse(bbox, fill=color)
            
            # Accent indicator (bright border)
            if is_accent and show_accents:
                draw.ellipse(
                    [x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2],
                    outline=(255, 255, 255),
                    width=2
                )
        
        # Draw border
        draw.rectangle(
            [margin_left, margin_top, width - margin_right, height - margin_bottom],
            outline=line_color,
            width=2
        )
        
        # Generate stats
        total_notes = len(events)
        visible_notes = len(visible_events)
        
        # Count by lane
        lane_counts = {}
        for e in events:
            lane = e.get("chart_lane_num", 1)
            lane_name = self.LANE_NAMES.get(lane, f"Lane {lane}")
            lane_counts[lane_name] = lane_counts.get(lane_name, 0) + 1
        
        cymbal_count = sum(1 for e in events if e.get("is_cymbal", False))
        accent_count = sum(1 for e in events if e.get("is_accent", False))
        ghost_count = sum(1 for e in events if e.get("is_ghost", False))
        
        stats = f"Total: {total_notes} notes | Visible: {visible_notes}\n"
        stats += f"Cymbals: {cymbal_count} | Accents: {accent_count} | Ghosts: {ghost_count}\n"
        stats += "By lane: " + ", ".join(f"{k}: {v}" for k, v in sorted(lane_counts.items()))
        
        # Convert to tensor for ComfyUI
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, C]
        
        print(f"[Drums2Chart] Preview: {visible_notes}/{total_notes} notes visible ({start_time:.1f}s - {end_time:.1f}s)")
        
        return (img_tensor, stats)


class PreviewChartAnimated:
    """
    Generate multiple preview frames for scrolling through the chart.
    
    Creates a batch of images showing different time windows.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "events": ("DRUM_EVENTS",),
                "audio": ("AUDIO",),  # To get duration
            },
            "optional": {
                "width": ("INT", {"default": 1280, "min": 640, "max": 4096}),
                "height": ("INT", {"default": 400, "min": 200, "max": 1080}),
                "seconds_per_frame": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 60.0}),
                "scroll_step": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.5,
                    "max": 30.0,
                    "tooltip": "Seconds to advance per frame"
                }),
                "dark_mode": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate_frames"
    CATEGORY = "audio/Drums2Chart"
    
    DESCRIPTION = "Generate scrolling preview frames of the entire chart."

    def generate_frames(
        self,
        events: List[Dict],
        audio: Dict[str, Any],
        width: int = 1280,
        height: int = 400,
        seconds_per_frame: float = 10.0,
        scroll_step: float = 5.0,
        dark_mode: bool = True,
    ) -> Tuple[torch.Tensor]:
        """Generate multiple preview frames"""
        
        # Get audio duration
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        duration = waveform.shape[-1] / sample_rate
        
        # Generate frames
        preview_node = PreviewChart()
        frames = []
        
        t = 0.0
        while t < duration:
            img_tensor, _ = preview_node.generate_preview(
                events=events,
                width=width,
                height=height,
                seconds_per_screen=seconds_per_frame,
                start_time=t,
                show_cymbal_rings=True,
                show_accents=True,
                show_ghosts=True,
                dark_mode=dark_mode,
            )
            frames.append(img_tensor)
            t += scroll_step
        
        # Stack frames
        if frames:
            result = torch.cat(frames, dim=0)  # [N, H, W, C]
        else:
            # Empty fallback
            result = torch.zeros(1, height, width, 3)
        
        print(f"[Drums2Chart] Generated {len(frames)} preview frames")
        
        return (result,)
