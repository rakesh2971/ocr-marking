
import cv2
import numpy as np
import re


class AnnotationFilter:
    def __init__(self):
        pass

    def detect_black_circles(self, image):
        """Detects black datum circles in the image. Returns list of (x, y, r)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=10, maxRadius=40
        )
        detected_circles = []
        if circles is not None:
            detected_circles = np.uint16(np.around(circles))
            detected_circles = detected_circles[0, :]
        return detected_circles

    def is_inside_circle(self, bbox, circles):
        """Returns True if the bbox centre lies inside any circle."""
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        for (cx, cy, r) in circles:
            dist = np.sqrt((center_x - cx) ** 2 + (center_y - cy) ** 2)
            if dist < r:
                return True
        return False

    def filter_by_circles(self, text_items, circles):
        """Filters out text items inside detected datum circles."""
        valid_items = []
        excluded_items = []
        if circles is None or len(circles) == 0:
            return text_items, []
        for item in text_items:
            if self.is_inside_circle(item['bbox'], circles):
                excluded_items.append(item)
            else:
                valid_items.append(item)
        return valid_items, excluded_items

    def filter_notes_section(self, text_items):
        """Dynamically finds the NOTES header and excludes text in that column."""
        notes_header = None
        for item in text_items:
            if item['text'].strip().upper() in ["NOTES", "NOTE"]:
                notes_header = item
                break
        if not notes_header:
            print("Warning: 'NOTES' header not found. Skipping notes filtering.")
            return text_items, []

        bbox = notes_header['bbox']
        header_x_min = min([p[0] for p in bbox])
        cutoff_x = header_x_min - 20

        valid_items = []
        excluded_items = []
        for item in text_items:
            item_bbox = item['bbox']
            item_center_x = sum(p[0] for p in item_bbox) / len(item_bbox)
            if item_center_x >= cutoff_x:
                excluded_items.append(item)
            else:
                valid_items.append(item)

        return valid_items, excluded_items

    def rescue_from_notes_filter(self, excluded_notes, all_raw_items):
        """
        Notes Rescue Pass: re-admits items whose left edge is clearly in the
        drawing area (more than 200px left of the NOTES header).
        Returns (rescued_list, still_excluded_list).
        """
        notes_header_x = None
        for ni in all_raw_items:
            if ni['text'].strip().upper() in ['NOTES', 'NOTE']:
                notes_header_x = min(p[0] for p in ni['bbox'])
                break
        if notes_header_x is None:
            return [], excluded_notes

        rescue_cutoff = notes_header_x - 200
        rescued = []
        still_excluded = []
        for ex_item in excluded_notes:
            item_x_min = min(p[0] for p in ex_item['bbox'])
            if item_x_min < rescue_cutoff:
                rescued.append(ex_item)
            else:
                still_excluded.append(ex_item)
        return rescued, still_excluded

    def rescue_gdt_items(self, all_excluded, existing_valid, clean_text_fn=None):
        """
        Generic GD&T Low-Confidence Rescue:
        Re-admits any filtered item whose normalised text matches the pattern
        X.XX[optional-letter] (e.g. '0.2 D', '1.2', '3.4').
        Dedup: skips if an existing valid item is already within 100px.
        
        Args:
            all_excluded: combined list of all excluded items across all filter passes
            existing_valid: current list of valid items (dedup reference)
            clean_text_fn: optional callable to normalise text (extractor.clean_text_content)
        Returns:
            list of newly rescued items
        """
        gdt_rescued = []
        for ex_item in all_excluded:
            raw_t = ex_item['text'].strip()
            norm = raw_t.replace('|E', ' D').replace('|e', ' D').replace('|d', ' D')
            norm = re.sub(r'(\d)\s+\.\s*(\d)', r'\1.\2', norm)
            if clean_text_fn:
                norm = clean_text_fn(norm).strip()
            if re.match(r'^\d+\.\d+\s*[A-Z]?$', norm) and len(norm) <= 8:
                ex_cx = sum(p[0] for p in ex_item['bbox']) / 4
                ex_cy = sum(p[1] for p in ex_item['bbox']) / 4
                already = any(
                    abs(sum(p[0] for p in v['bbox']) / 4 - ex_cx) < 100 and
                    abs(sum(p[1] for p in v['bbox']) / 4 - ex_cy) < 100
                    for v in existing_valid
                )
                if not already:
                    copy = dict(ex_item)
                    copy['text'] = norm
                    gdt_rescued.append(copy)
                    print(f"  - GD&T rescue: '{norm}' (was '{raw_t}')")
        return gdt_rescued

    def filter_bottom_right_table(self, text_items, image_shape):
        """Filters out text in the bottom-right parts/material table."""
        if not text_items:
            return text_items, []
        h, w = image_shape[:2]
        search_x_min = w / 2
        search_y_min = h / 2
        anchors = ["TABLE NO", "COATING WEIGHT", "MATERAIL", "MATERIAL", "ITEM PART NO", "DESCRIPTION"]
        table_top_y = h
        table_left_x = w
        found_anchor = False
        for item in text_items:
            text = item['text'].strip().upper()
            bbox = item['bbox']
            x_min = min([p[0] for p in bbox])
            y_min = min([p[1] for p in bbox])
            if x_min > search_x_min and y_min > search_y_min:
                is_anchor = any(anchor in text for anchor in anchors)
                if is_anchor:
                    found_anchor = True
                    if y_min < table_top_y:
                        table_top_y = y_min
                    if x_min < table_left_x:
                        table_left_x = x_min
        if not found_anchor:
            print("Warning: Table anchors not found in bottom-right. Skipping table filtering.")
            return text_items, []
        cutoff_y = table_top_y - 20
        cutoff_x = table_left_x - 20
        valid_items = []
        excluded_items = []
        for item in text_items:
            bbox = item['bbox']
            x_min = min([p[0] for p in bbox])
            y_min = min([p[1] for p in bbox])
            if x_min >= cutoff_x and y_min >= cutoff_y:
                excluded_items.append(item)
            else:
                valid_items.append(item)
        return valid_items, excluded_items

    def filter_view_labels(self, text_items):
        """Filters view labels (SECTION, VIEW, FP56, etc.) unless they contain dimensions."""
        valid_items = []
        excluded_items = []
        keywords = ["VIEW", "SECTION", "DETAIL", "FRONT", "SIDE", "PT6", "PT3", "FP4", "FP3", "FP56"]
        dimension_pattern = re.compile(r'\d+\.\d+|X\s*Y\s*Z|X\s*Y|\bX\b|\bY\b|\bZ\b')
        for item in text_items:
            text = item['text'].upper()
            is_view_label = any(kw in text for kw in keywords)
            if is_view_label:
                has_dimensions = dimension_pattern.search(text) is not None
                if has_dimensions:
                    valid_items.append(item)
                else:
                    excluded_items.append(item)
            else:
                valid_items.append(item)
        return valid_items, excluded_items

    def filter_top_left_numbers(self, text_items, image_shape):
        """Filters garbage sheet numbers in the extreme top-left corner."""
        if not text_items:
            return text_items, []
        h, w = image_shape[:2]
        valid_items = []
        excluded_items = []
        for item in text_items:
            bbox = item['bbox']
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            center_x = sum(x_coords) / 4
            center_y = sum(y_coords) / 4
            if center_x < (w * 0.25) and center_y < (h * 0.035):
                excluded_items.append(item)
            else:
                valid_items.append(item)
        return valid_items, excluded_items
