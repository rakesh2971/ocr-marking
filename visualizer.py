
import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    def draw_annotations(self, image, text_items, start_id=1):
        """
        Draws red rectangles and sequential numbers around valid text items.
        Skips DATUM_BUBBLE items entirely (no box, no number, no counter increment).
        Returns the annotated image and the list of mapping dicts.
        """
        annotated_image = image.copy()
        mappings = []
        idx = start_id

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

            circle_radius = 25
            
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
            
            # Convert image to grayscale for white-space checking if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
                
            img_h, img_w = gray_image.shape
            
            # Keep track of placed balloons so we don't overlap them
            if not hasattr(self, '_placed_balloons'):
                self._placed_balloons = []
                
            for dist in [50, 75, 100, 125, 150]:
                for angle in angles_to_try:
                    cx = int(line_start_x + dist * np.cos(angle))
                    cy = int(line_start_y + dist * np.sin(angle))
                    
                    # Ensure the whole balloon is within the image bounds
                    if (cx - circle_radius < 0 or cx + circle_radius >= img_w or
                        cy - circle_radius < 0 or cy + circle_radius >= img_h):
                        continue
                        
                    # Check collision with existing balloons
                    collision = False
                    for (px, py) in self._placed_balloons:
                        # Distance between centers must be > 2 * radius + buffer
                        if np.sqrt((cx - px)**2 + (cy - py)**2) < (circle_radius * 2 + 10):
                            collision = True
                            break
                    if collision:
                        continue
                        
                    # Evaluate underlying image pixels for "emptiness" (lack of black lines)
                    # Create a mask for the proposed circle area
                    y_idx, x_idx = np.ogrid[:img_h, :img_w]
                    mask = ((x_idx - cx)**2 + (y_idx - cy)**2 <= (circle_radius + 5)**2)
                    
                    # Count "dark" pixels (indicating drawing lines or text)
                    # Threshold for "dark" is e.g. < 200 out of 255
                    roi_pixels = gray_image[mask]
                    dark_pixels = np.sum(roi_pixels < 200)
                    
                    if dark_pixels < lowest_dark_pixels:
                        lowest_dark_pixels = dark_pixels
                        best_cx, best_cy = cx, cy
                        
                    # If we find a spot that is almost entirely white, take it immediately
                    if dark_pixels < 10:
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
            font_scale = 1.0
            thickness = 2
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
