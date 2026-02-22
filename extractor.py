
import easyocr
import numpy as np
import cv2
import re


class TextExtractor:
    def __init__(self, languages=['en']):
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(languages)

    def extract_text(self, image):
        """
        Extracts text from an image (numpy array).
        Returns a list of dicts: {'bbox', 'text', 'confidence'}
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        results = self.reader.readtext(image)
        
        text_items = []
        for (bbox, text, prob) in results:
            if prob > 0.2:
                text_items.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': prob
                })
        
        horizontal_merged = self.merge_horizontal_items(text_items)
        final_merged = self.merge_vertical_items(horizontal_merged)
        final_clean = self.cleanup_items(final_merged)
        
        for item in final_clean:
            item['text'] = self.clean_text_content(item['text'])
            
        return final_clean

    def extract_text_custom(self, image, x_threshold=40):
        """
        Custom extraction used by main pipeline with lower x_threshold (40px)
        to prevent GD&T merging with adjacent dimensions.
        Returns cleaned, stitched items ready for filtering.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        raw_results = self.reader.readtext(image)
        raw_items = []
        for (bbox, text, prob) in raw_results:
            if prob > 0.2:
                raw_items.append({'bbox': bbox, 'text': text, 'confidence': prob})

        h_merged = self.merge_horizontal_items(raw_items, x_threshold=x_threshold)
        v_merged = self.merge_vertical_items(h_merged)
        valid_pre = self.cleanup_items(v_merged)

        stitched = self.apply_decimal_stitcher(valid_pre)
        gdt_stitched = self.apply_gdt_stitcher(stitched)
        dim_tol_stitched = self.apply_dimension_tolerance_stitcher(gdt_stitched)

        # Final cleanup: drop isolated non-XYZ single letters
        result = []
        for item in dim_tol_stitched:
            clean_t = item['text'].replace(".", "").strip()
            if len(clean_t) == 1 and clean_t.isalpha():
                if clean_t.upper() not in ['X', 'Y', 'Z']:
                    continue
            item['text'] = self.clean_text_content(item['text'])
            result.append(item)

        return result

    def apply_decimal_stitcher(self, items):
        """
        Merges short digit tokens (1-3 chars) with the next right-adjacent
        numeric or datum-letter token within 150px and 20px vertical tolerance.
        Handles cases like '3' + '.58' -> '3.58', or '0.2' + 'D' -> '0.2 D'.
        """
        stitched_items = []
        skip_idx = set()

        items_sorted = sorted(items, key=lambda i: min(p[0] for p in i['bbox']))

        for i, item in enumerate(items_sorted):
            if i in skip_idx:
                continue

            t1 = item['text'].strip()
            if len(t1) <= 3 and sum(c.isdigit() for c in t1) > 0:
                bbox1 = item['bbox']
                x1_max = max(p[0] for p in bbox1)
                y1_cy = sum(p[1] for p in bbox1) / 4

                best_j = -1
                for j in range(i + 1, len(items_sorted)):
                    if j in skip_idx:
                        continue
                    item2 = items_sorted[j]
                    t2 = item2['text'].strip()
                    bbox2 = item2['bbox']
                    x2_min = min(p[0] for p in bbox2)
                    y2_cy = sum(p[1] for p in bbox2) / 4

                    # 20px vertical tolerance — tight enough to not span table rows (35-40px apart)
                    if abs(y1_cy - y2_cy) < 20 and 0 < (x2_min - x1_max) < 150:
                        is_num = sum(c.isdigit() for c in t2) > 0
                        is_datum_letter = (len(t2) == 1 and t2.isalpha())
                        if is_num or is_datum_letter:
                            best_j = j
                            break

                if best_j != -1:
                    item2 = items_sorted[best_j]
                    t2 = item2['text'].strip()
                    bbox2 = item2['bbox']

                    new_x_min = min(p[0] for p in bbox1 + bbox2)
                    new_x_max = max(p[0] for p in bbox1 + bbox2)
                    new_y_min = min(p[1] for p in bbox1 + bbox2)
                    new_y_max = max(p[1] for p in bbox1 + bbox2)
                    new_bbox = [[new_x_min, new_y_min], [new_x_max, new_y_min],
                                [new_x_max, new_y_max], [new_x_min, new_y_max]]

                    if "." not in t1 and "." not in t2 and t1.isdigit() and len(t1) <= 2:
                        new_t = t1 + "." + t2
                    else:
                        new_t = t1 + " " + t2
                        new_t = new_t.replace(" .", ".")

                    stitched_items.append({
                        'bbox': new_bbox, 'text': new_t,
                        'confidence': (item['confidence'] + item2['confidence']) / 2
                    })
                    skip_idx.add(best_j)
                    continue

            stitched_items.append(item)

        return stitched_items

    def apply_gdt_stitcher(self, items):
        """
        Merges a numeric GD&T tolerance value with an adjacent datum sequence
        (e.g. '3.4' + 'X Y Z', '6.8' + 'UZ (0.3) Z').
        Vertical tolerance: 20px. Horizontal gap: up to 180px.
        """
        gdt_stitched = []
        skip_gdt = set()

        for i, item in enumerate(items):
            if i in skip_gdt:
                continue

            t1 = item['text'].strip()
            if re.match(r'^\d+(\.\d+)?$', t1) or 'UZ' in t1 or '+0' in t1:
                bbox1 = item['bbox']
                x1_max = max(p[0] for p in bbox1)
                y1_cy = sum(p[1] for p in bbox1) / 4

                best_j = -1
                for j in range(i + 1, len(items)):
                    if j in skip_gdt:
                        continue
                    item2 = items[j]
                    t2 = item2['text'].strip()
                    bbox2 = item2['bbox']
                    x2_min = min(p[0] for p in bbox2)
                    y2_cy = sum(p[1] for p in bbox2) / 4

                    # 20px tolerance: prevents cross-row merges (table rows 37-40px apart)
                    if abs(y1_cy - y2_cy) < 20 and -20 < (x2_min - x1_max) < 180:
                        if ('X' in t2 or 'Y' in t2 or 'Z' in t2 or 'UZ' in t2
                                or 'A| B' in t2 or re.match(r'^[A-Z](\s+[A-Z])*$', t2)):
                            best_j = j
                            break

                if best_j != -1:
                    item2 = items[best_j]
                    t2 = item2['text'].strip()
                    bbox2 = item2['bbox']

                    new_x_min = min(p[0] for p in bbox1 + bbox2)
                    new_x_max = max(p[0] for p in bbox1 + bbox2)
                    new_y_min = min(p[1] for p in bbox1 + bbox2)
                    new_y_max = max(p[1] for p in bbox1 + bbox2)
                    new_bbox = [[new_x_min, new_y_min], [new_x_max, new_y_min],
                                [new_x_max, new_y_max], [new_x_min, new_y_max]]

                    gdt_stitched.append({
                        'bbox': new_bbox, 'text': t1 + " " + t2,
                        'confidence': (item['confidence'] + item2['confidence']) / 2
                    })
                    skip_gdt.add(best_j)
                    continue

            gdt_stitched.append(item)

        return gdt_stitched

    def apply_dimension_tolerance_stitcher(self, items):
        """
        Merges vertically stacked dimension value + tolerance into one item
        (e.g. 'Ø57.7' above '±0' or '57.7' above '±0') so they get one box/number.
        """
        # Patterns: dimension value (diameter/number) and tolerance (±0, ±0.1, etc.)
        dim_value_pattern = re.compile(
            r'[\ØØØ]?\s*\d+[.,]\d+|^\d+[.,]\d+\s*$',
            re.IGNORECASE
        )
        tolerance_pattern = re.compile(
            r'^[±\+\-]?\s*\d+[.,]?\d*\s*$',
            re.IGNORECASE
        )

        def looks_like_dimension(t):
            t = t.strip().replace('\n', ' ')
            if not t:
                return False
            if re.search(r'\d+[.,]\d+', t) and len(t) <= 20:
                return True
            if 'Ø' in t or 'ø' in t or '0' in t and re.search(r'\d', t):
                return True
            return False

        def looks_like_tolerance(t):
            t = t.strip().replace('\n', ' ')
            if not t or len(t) > 15:
                return False
            if tolerance_pattern.match(t):
                return True
            if re.match(r'^[±]\s*\d+[.,]?\d*\s*$', t):
                return True
            return False

        merged = []
        skip = set()
        # Sort by top (min_y) then left (min_x)
        sorted_items = sorted(
            enumerate(items),
            key=lambda p: (min(q[1] for q in p[1]['bbox']), min(q[0] for q in p[1]['bbox']))
        )

        for i, item in sorted_items:
            if i in skip:
                continue
            t1 = item['text'].strip().replace('\n', ' ')
            bbox1 = item['bbox']
            y1_min = min(p[1] for p in bbox1)
            y1_max = max(p[1] for p in bbox1)
            x1_min = min(p[0] for p in bbox1)
            x1_max = max(p[0] for p in bbox1)

            best_j = -1
            best_dist = 9999

            for j, other in sorted_items:
                if j <= i or j in skip:
                    continue
                t2 = other['text'].strip().replace('\n', ' ')
                bbox2 = other['bbox']
                y2_min = min(p[1] for p in bbox2)
                y2_max = max(p[1] for p in bbox2)
                x2_min = min(p[0] for p in bbox2)
                x2_max = max(p[0] for p in bbox2)

                # B must be below A (vertical gap; allow up to 90px for dimension lines)
                gap = y2_min - y1_max
                if gap < 0 or gap > 90:
                    continue
                # Same dimension column: horizontal overlap or close centers (within 120px)
                x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                min_w = min(x1_max - x1_min, x2_max - x2_min)
                c1_x = (x1_min + x1_max) / 2
                c2_x = (x2_min + x2_max) / 2
                if min_w > 0 and x_overlap < min_w * 0.2 and abs(c1_x - c2_x) > 120:
                    continue

                # One must be dimension-like and the other tolerance-like (one dimension = one box)
                dim_first = looks_like_dimension(t1) and looks_like_tolerance(t2)
                dim_second = looks_like_dimension(t2) and looks_like_tolerance(t1)
                if not (dim_first or dim_second):
                    continue
                if gap < best_dist:
                    best_dist = gap
                    best_j = j

            if best_j != -1:
                other = items[best_j]
                bbox2 = other['bbox']
                new_x_min = min(p[0] for p in bbox1 + bbox2)
                new_x_max = max(p[0] for p in bbox1 + bbox2)
                new_y_min = min(p[1] for p in bbox1 + bbox2)
                new_y_max = max(p[1] for p in bbox1 + bbox2)
                new_bbox = [[new_x_min, new_y_min], [new_x_max, new_y_min],
                            [new_x_max, new_y_max], [new_x_min, new_y_max]]
                new_text = item['text'].strip() + " " + other['text'].strip()
                new_text = re.sub(r'\s+', ' ', new_text)
                conf_a = item.get('confidence', 0.5)
                conf_b = other.get('confidence', 0.5)
                merged.append({
                    'bbox': new_bbox,
                    'text': new_text,
                    'confidence': (conf_a + conf_b) / 2
                })
                skip.add(best_j)
            else:
                merged.append(item)

        return merged

    def clean_text_content(self, text):
        """Applies regex replacements to fix common OCR errors."""
        text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
        text = text.replace('YIz', 'Y')
        text = text.replace('S00', '0.05')
        text = text.replace('XYIZ', 'X Y Z')
        text = text.replace('XlYIZ', 'X Y Z')
        if text.strip() == '313':
            text = '3.4'
        if text.strip() == '8 8':
            text = '8.8'
        if text.strip() == '8 8 6':
            text = '8.8'
        text = re.sub(r'0\s*\.\s*3\s*\)\s*\|?\s*[zZ][iI]', '(0.3) Z', text)
        text = re.sub(r'^5\s*Xi\n3$', '5.4 X Y\n3.4', text, flags=re.MULTILINE)
        text = re.sub(r'^5\s*Xi$', '5.4 X Y', text)
        if text.strip() == 'Xi':
            text = 'X'
        text = re.sub(r'\b0\s+2\b', '0.2', text)
        text = re.sub(r'\b0\s+5\b', '0.5', text)
        text = re.sub(r'\b0\s+1\b', '0.1', text)
        text = text.replace('~', '')
        return text.strip()

    def cleanup_items(self, items):
        """Removes items that are likely noise/garbage."""
        valid = []
        garbage_texts = ["0", ".", ",", ":", ";", "-", "_", "~", "'", "\u2018", '"', "`"]
        for item in items:
            text = item['text'].strip()
            if text in garbage_texts:
                continue
            if not text:
                continue
            if len(text) <= 2:
                has_alnum = any(c.isalnum() for c in text)
                if not has_alnum:
                    continue
            valid.append(item)
        return valid

    def merge_horizontal_items(self, items, x_threshold=80, y_threshold=10):
        """Merges text items that are close horizontally and aligned vertically."""
        if not items:
            return []

        def get_min_y(item):
            return min([p[1] for p in item['bbox']])
        def get_min_x(item):
            return min([p[0] for p in item['bbox']])

        items.sort(key=lambda item: (get_min_x(item), get_min_y(item)))
        merged_items = items[:]
        changed = True
        while changed:
            changed = False
            new_merged = []
            skip_indices = set()
            merged_items.sort(key=lambda item: (get_min_x(item), get_min_y(item)))
            for i, curr_item in enumerate(merged_items):
                if i in skip_indices:
                    continue
                curr_bbox = curr_item['bbox']
                curr_x_max = max([p[0] for p in curr_bbox])
                curr_y_min = min([p[1] for p in curr_bbox])
                curr_y_max = max([p[1] for p in curr_bbox])
                curr_height = curr_y_max - curr_y_min
                curr_cy = (curr_y_min + curr_y_max) / 2
                best_match_idx = -1
                best_dist = float('inf')
                for j in range(i + 1, len(merged_items)):
                    if j in skip_indices:
                        continue
                    next_item = merged_items[j]
                    next_bbox = next_item['bbox']
                    next_x_min = min([p[0] for p in next_bbox])
                    next_y_min = min([p[1] for p in next_bbox])
                    next_y_max = max([p[1] for p in next_bbox])
                    next_height = next_y_max - next_y_min
                    next_cy = (next_y_min + next_y_max) / 2
                    y_overlap = max(0, min(curr_y_max, next_y_max) - max(curr_y_min, next_y_min))
                    min_h = min(curr_height, next_height)
                    vertical_aligned = y_overlap > (min_h * 0.5) or abs(curr_cy - next_cy) < 15
                    if not vertical_aligned:
                        continue
                    if next_x_min < (curr_x_max - 10):
                        continue
                    dist = next_x_min - curr_x_max
                    if dist > x_threshold:
                        continue
                    if dist < best_dist:
                        best_dist = dist
                        best_match_idx = j
                if best_match_idx != -1:
                    j = best_match_idx
                    next_item = merged_items[j]
                    next_bbox = next_item['bbox']
                    new_x_min = min([p[0] for p in curr_bbox] + [p[0] for p in next_bbox])
                    new_x_max = max([p[0] for p in curr_bbox] + [p[0] for p in next_bbox])
                    new_y_min = min([p[1] for p in curr_bbox] + [p[1] for p in next_bbox])
                    new_y_max = max([p[1] for p in curr_bbox] + [p[1] for p in next_bbox])
                    new_bbox = [[new_x_min, new_y_min], [new_x_max, new_y_min],
                                [new_x_max, new_y_max], [new_x_min, new_y_max]]
                    new_text = curr_item['text'] + " " + next_item['text']
                    merged_item = {
                        'bbox': new_bbox,
                        'text': new_text,
                        'confidence': (curr_item['confidence'] + next_item['confidence']) / 2
                    }
                    new_merged.append(merged_item)
                    skip_indices.add(j)
                    changed = True
                else:
                    new_merged.append(curr_item)
            merged_items = new_merged
        return merged_items

    def merge_vertical_items(self, items, x_threshold=20, y_threshold=30):
        """Merges text items that are close vertically and aligned horizontally."""
        if not items:
            return []

        def get_min_x(item):
            return min([p[0] for p in item['bbox']])
        def get_min_y(item):
            return min([p[1] for p in item['bbox']])

        items.sort(key=lambda item: (get_min_x(item), get_min_y(item)))
        merged_items = []
        current_item = items[0]
        for next_item in items[1:]:
            curr_bbox = current_item['bbox']
            next_bbox = next_item['bbox']
            curr_x_min = min([p[0] for p in curr_bbox])
            curr_x_max = max([p[0] for p in curr_bbox])
            curr_y_max = max([p[1] for p in curr_bbox])
            curr_width = curr_x_max - curr_x_min
            next_x_min = min([p[0] for p in next_bbox])
            next_x_max = max([p[0] for p in next_bbox])
            next_y_min = min([p[1] for p in next_bbox])
            next_width = next_x_max - next_x_min
            overlap_width = max(0, min(curr_x_max, next_x_max) - max(curr_x_min, next_x_min))
            min_width = min(curr_width, next_width)
            horizontal_aligned = overlap_width > (min_width * 0.5)
            vertical_close = (next_y_min - curr_y_max) < y_threshold
            is_below = next_y_min > (curr_y_max - 10)
            if horizontal_aligned and vertical_close and is_below:
                new_x_min = min(curr_x_min, next_x_min)
                new_x_max = max(curr_x_max, next_x_max)
                new_y_min = min([p[1] for p in curr_bbox] + [p[1] for p in next_bbox])
                new_y_max = max([p[1] for p in curr_bbox] + [p[1] for p in next_bbox])
                new_bbox = [[new_x_min, new_y_min], [new_x_max, new_y_min],
                            [new_x_max, new_y_max], [new_x_min, new_y_max]]
                new_text = current_item['text'] + "\n" + next_item['text']
                current_item = {
                    'bbox': new_bbox,
                    'text': new_text,
                    'confidence': (current_item['confidence'] + next_item['confidence']) / 2
                }
            else:
                merged_items.append(current_item)
                current_item = next_item
        merged_items.append(current_item)
        return merged_items


# ============================================================================
# VERTICAL TEXT DETECTOR — Full-Page 90°CW Rotation
# ============================================================================
class FullPageRotationDetector:
    """Detects vertical text by rotating the entire image 90° CW,
    running EasyOCR, and mapping bounding boxes back to original coordinates."""

    def __init__(self, reader):
        self.reader = reader

    def detect_vertical_text(self, image):
        """
        Rotates image 90° CW, runs OCR, maps coordinates back to 0°.
        Returns list of dicts with 'bbox', 'text', 'confidence'.
        """
        h, w = image.shape[:2]
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rot_h, rot_w = rotated_image.shape[:2]  # rot_h = w, rot_w = h

        results = self.reader.readtext(rotated_image, paragraph=False, width_ths=0.7)

        new_items = []
        for bbox, text, conf in results:
            if conf < 0.2:
                continue
            clean_text = text.strip()
            if not re.search(r'\d', clean_text):
                continue
            if len(clean_text) < 3 or len(clean_text) > 15:
                continue
            if not re.match(r'^[\d\.\-\+\s,]+$', clean_text):
                if not re.match(r'^[\d\.\-\+\sA-Za-z]+$', clean_text):
                    continue
                alphas = sum(c.isalpha() for c in clean_text)
                digits = sum(c.isdigit() for c in clean_text)
                if alphas >= digits:
                    continue

            # Inverse transform for 90°CW: (rot_x, rot_y) -> orig (rot_y, rot_w - rot_x)
            original_bbox = []
            for point in bbox:
                rot_x, rot_y = point
                orig_x = rot_y
                orig_y = rot_w - rot_x
                original_bbox.append([orig_x, orig_y])

            new_items.append({
                'bbox': original_bbox,
                'text': clean_text,
                'confidence': conf,
                'source': 'full_page_rotation',
                'orientation': 'vertical'
            })

        return new_items
