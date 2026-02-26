
import cv2
import numpy as np
from config import (
    BALLOON_RADIUS,
    BALLOON_DISTANCES,
    BALLOON_DARK_THRESHOLD,
    BALLOON_FONT_SCALE,
    BALLOON_THICKNESS
)

class Visualizer:
    def __init__(self):
        pass

    def draw_annotations(self, image, text_items, start_id=1, gray_image=None):
        """
        Draws red rectangles and sequential numbers around valid text items.
        Skips DATUM_BUBBLE items entirely (no box, no number, no counter increment).
        Returns the annotated image and the list of mapping dicts.
        """
        annotated_image = image.copy()
        mappings = []
        idx = start_id
        self._placed_balloons = []   # reset every call so page 2+ don't avoid page 1 positions

        for item in text_items:
            # Skip datum bubbles — they are reference markers, not annotations
            if item.get('type') == 'DATUM_BUBBLE':
                continue

            bbox = item['bbox']
            text = item['text']

            x_min = int(min([p[0] for p in bbox]))
            x_max = int(max([p[0] for p in bbox]))
            y_min = int(min([p[1] for p in bbox]))
            y_max = int(max([p[1] for p in bbox]))

            # Box styling — image is RGB (from fitz/PyMuPDF), so Red = (255, 0, 0)
            box_color = (255, 0, 0) # Red in RGB
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), box_color, 2)

            circle_radius = BALLOON_RADIUS
            
            # Start of the balloon line (top middle or top left of the bounding box)
            line_start_x = x_min + int((x_max - x_min) * 0.25)
            line_start_y = y_min
            
            # --- Spatial search for an empty spot ---
            # We want to place the balloon where it doesn't cover drawing details
            # and doesn't overlap other balloons. We try multiple angles and distances.
            best_cx, best_cy = line_start_x - 20, line_start_y - 60 # Default fallback
            lowest_dark_pixels = float('inf')
            found_clear_spot = False
            
            # Check 8 directions (45 deg increments), distances from 50 to 150 px
            angles_to_try = [
                -np.pi/2,         # Straight up (preferred)
                -np.pi/2 - np.pi/4, # Up-left
                -np.pi/2 + np.pi/4, # Up-right
                np.pi,            # Left
                0,                # Right
                np.pi/4,          # Down-right
                np.pi - np.pi/4,  # Down-left
                np.pi/2           # Straight down
            ]
            
            # Use the pre-computed grayscale image for white-space checking
            if gray_image is None:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                
            img_h, img_w = gray_image.shape
            

            for dist in BALLOON_DISTANCES:
                for angle in angles_to_try:
                    cx = int(line_start_x + dist * np.cos(angle))
                    cy = int(line_start_y + dist * np.sin(angle))
                    
                    # Ensure the whole balloon is within the image bounds
                    if (cx - circle_radius < 0 or cx + circle_radius >= img_w or
                        cy - circle_radius < 0 or cy + circle_radius >= img_h):
                        continue
                        
                    # Check collision with existing balloons (vectorized for speed)
                    collision = False
                    if len(self._placed_balloons) > 0:
                        pts = np.array(self._placed_balloons)
                        dist_sq = (pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2
                        min_dist_allowed_sq = (circle_radius * 2 + 10)**2
                        if np.any(dist_sq < min_dist_allowed_sq):
                            continue
                            
                    # Evaluate underlying image pixels for "emptiness" (lack of black lines)
                    y1 = max(0, cy - circle_radius - 5)
                    y2 = min(img_h, cy + circle_radius + 5)
                    x1 = max(0, cx - circle_radius - 5)
                    x2 = min(img_w, cx + circle_radius + 5)
                    
                    if y2 <= y1 or x2 <= x1:
                        continue
                        
                    roi_pixels = gray_image[y1:y2, x1:x2]
                    roi_h, roi_w = roi_pixels.shape
                    
                    y_idx, x_idx = np.ogrid[:roi_h, :roi_w]
                    local_cy, local_cx = cy - y1, cx - x1
                    mask = ((x_idx - local_cx)**2 + (y_idx - local_cy)**2 <= (circle_radius + 5)**2)
                    
                    # Count "dark" pixels (indicating drawing lines or text)
                    dark_pixels = np.sum(roi_pixels[mask] < 200)
                    
                    if dark_pixels < lowest_dark_pixels:
                        lowest_dark_pixels = dark_pixels
                        best_cx, best_cy = cx, cy
                        
                    # If we find a spot that is almost entirely white, take it immediately
                    if dark_pixels < BALLOON_DARK_THRESHOLD:
                        found_clear_spot = True
                        break
                if found_clear_spot:
                    break
                    
            circle_center_x, circle_center_y = best_cx, best_cy
            self._placed_balloons.append((circle_center_x, circle_center_y))
            # ----------------------------------------

            circle_center = (circle_center_x, circle_center_y)

            # Draw connection line
            # Draw line from center to center, but we don't draw over the circle itself
            cv2.line(annotated_image, circle_center, (line_start_x, line_start_y), box_color, 2)

            # Draw circle (white filled interior, red outline) - this covers the start of the line
            cv2.circle(annotated_image, circle_center, circle_radius, (255, 255, 255), -1) # White fill
            cv2.circle(annotated_image, circle_center, circle_radius, box_color, 2)        # Red outline

            # Draw Number
            label = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Scale font nicely within circle
            font_scale = BALLOON_FONT_SCALE
            thickness = BALLOON_THICKNESS
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Centering text inside circle
            text_x = circle_center[0] - int(text_w / 2)
            text_y = circle_center[1] + int(text_h / 2)
            
            # Red text
            cv2.putText(annotated_image, label, (text_x, text_y),
                        font, font_scale, box_color, thickness)

            mappings.append({
                'id': idx,
                'description': text,
                'type': item.get('type', 'OTHER')
            })
            idx += 1

        return annotated_image, mappings

    def draw_debug_circles(self, image, circles):
        """
        Draws detected exclusion circles in Green for debugging.
        """
        debug_image = image.copy()
        if circles is not None:
            for (x, y, r) in circles:
                cv2.circle(debug_image, (x, y), r, (0, 255, 0), 2)
        return debug_image
