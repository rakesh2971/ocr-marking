"""
_legacy_stitchers.py

DEPRECATED: These stitchers were originally written to reconstruct fragmented 
text blocks when the pipeline relied exclusively on EasyOCR.

PaddleOCR natively handles block-level grouping much better, making these 
post-processing steps largely obsolete. They are kept here strictly for 
reference or fallback implementation.

DO NOT call these from active pipeline code (`pipeline.py` or `main.py`).
"""
import re

class LegacyStitchers:
    def apply_decimal_stitcher(self, items):
        """
        Merges strictly classified integer and decimal fragments based on semantics.
        Handles cases like '3' (INTEGER) + '.58' (DECIMAL_FRAGMENT) -> '3.58'
        """
        stitched_items = []
        skip_idx = set()

        # Pre-compute x_min for O(n) memory rather than O(n log n) loop recalculation
        keyed = [(min(p[0] for p in it['bbox']), it) for it in items]
        keyed.sort(key=lambda t: t[0])
        items_sorted = [it for _, it in keyed]

        for i, item in enumerate(items_sorted):
            if i in skip_idx:
                continue

            if item.get("type") == "INTEGER":
                bbox1 = item['bbox']
                x1_max = max(p[0] for p in bbox1)
                y1_cy = sum(p[1] for p in bbox1) / 4

                best_j = -1
                for j in range(i + 1, len(items_sorted)):
                    if j in skip_idx:
                        continue
                    item2 = items_sorted[j]
                    bbox2 = item2['bbox']
                    x2_min = min(p[0] for p in bbox2)
                    y2_cy = sum(p[1] for p in bbox2) / 4

                    # 20px vertical tolerance
                    if abs(y1_cy - y2_cy) < 20 and 0 <= (x2_min - x1_max) < 150:
                        if item2.get("type") in ["DECIMAL_FRAGMENT", "DECIMAL_PREFIX"]:
                            best_j = j
                            break

                if best_j != -1:
                    item2 = items_sorted[best_j]
                    bbox2 = item2['bbox']

                    new_x_min = min(p[0] for p in bbox1 + bbox2)
                    new_x_max = max(p[0] for p in bbox1 + bbox2)
                    new_y_min = min(p[1] for p in bbox1 + bbox2)
                    new_y_max = max(p[1] for p in bbox1 + bbox2)
                    new_bbox = [[new_x_min, new_y_min], [new_x_max, new_y_min],
                                [new_x_max, new_y_max], [new_x_min, new_y_max]]

                    new_t = item['text'].strip() + item2['text'].strip()
                    
                    new_item = {
                        'bbox': new_bbox, 'text': new_t,
                        'confidence': (item['confidence'] + item2['confidence']) / 2,
                        'type': self.classify_token(new_t) # assumes subclassed or patched
                    }
                    stitched_items.append(new_item)
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
        # Pre-compute (min_y, min_x)
        keyed_items = [
            ((min(q[1] for q in it['bbox']), min(q[0] for q in it['bbox'])), i, it)
            for i, it in enumerate(items)
        ]
        keyed_items.sort(key=lambda x: x[0])
        sorted_items = [(idx, it) for _, idx, it in keyed_items]

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
