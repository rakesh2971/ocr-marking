
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
            # ── Skip datum bubbles — they are reference markers, not annotations ──
            if item.get('type') == 'DATUM_BUBBLE':
                continue

            bbox = item['bbox']
            text = item['text']

            x_min = int(min([p[0] for p in bbox]))
            x_max = int(max([p[0] for p in bbox]))
            y_min = int(min([p[1] for p in bbox]))
            y_max = int(max([p[1] for p in bbox]))

            # Draw Red Rectangle
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Draw Number Circle at top-left
            circle_radius = 15
            circle_center = (x_min, y_min)
            cv2.circle(annotated_image, circle_center, circle_radius, (255, 0, 0), -1)

            label = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = circle_center[0] - int(text_w / 2)
            text_y = circle_center[1] + int(text_h / 2)
            cv2.putText(annotated_image, label, (text_x, text_y),
                        font, font_scale, (255, 255, 255), thickness)

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
