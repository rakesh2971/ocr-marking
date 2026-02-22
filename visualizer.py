
import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    def draw_annotations(self, image, text_items, start_id=1):
        """
        Draws red circles and numbers around the valid text items.
        Returns the annotated image and the list of (number, description) tuples.
        """
        annotated_image = image.copy()
        mappings = []
        
        # Draw a blue bounding box around the entire cluster to show grouping
        if text_items:
            all_x = [p[0] for item in text_items for p in item['bbox']]
            all_y = [p[1] for item in text_items for p in item['bbox']]
            if all_x and all_y:
                cx_min, cx_max = int(min(all_x)), int(max(all_x))
                cy_min, cy_max = int(min(all_y)), int(max(all_y))
                # Image is RGB from PyMuPDF, so Blue is (0, 0, 255)
                cv2.rectangle(annotated_image, (cx_min, cy_min), (cx_max, cy_max), (0, 0, 255), 2)
        
        # Helper to get numeric sort key if needed, but we'll specific sequential ID
        # For now, just iterate and assign 1, 2, 3...
        
        for idx, item in enumerate(text_items, start=start_id):
            bbox = item['bbox']
            text = item['text']
            
            # Calculate center and radius for drawing the framing circle
            # bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            # Cast to int to ensure OpenCV compatibility
            x_min = int(min([p[0] for p in bbox]))
            x_max = int(max([p[0] for p in bbox]))
            y_min = int(min([p[1] for p in bbox]))
            y_max = int(max([p[1] for p in bbox]))
            
            # Draw Red Rectangle (Box)
            # Image is RGB from PyMuPDF, so Red is (255, 0, 0)
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Draw Number Circle
            # Attach it to the top-left corner
            circle_radius = 15
            circle_center = (x_min, y_min)
            
            # Draw filled red circle for the number background
            cv2.circle(annotated_image, circle_center, circle_radius, (255, 0, 0), -1)
            
            # Text properties
            label = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Calculate text size to center it
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = circle_center[0] - int(text_w / 2)
            text_y = circle_center[1] + int(text_h / 2) # Adjust for baseline
            
            # Draw White Number
            cv2.putText(annotated_image, label, (text_x, text_y), 
                        font, font_scale, (255, 255, 255), thickness)
            
            mappings.append({
                'id': idx,
                'description': text
            })
            
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
