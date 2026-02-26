"""
box_detector.py - Detects isolated single-character boxes (datum/reference markers)
in engineering drawings using OpenCV contour detection + targeted OCR.

These are standalone small rectangles containing characters like X, E, Z, Z1, B
that EasyOCR's full-page scan often misses.
"""
import cv2
import numpy as np
import re
from config import (
    BOX_AREA_MIN, BOX_AREA_MAX, BOX_ASPECT_MIN, BOX_ASPECT_MAX,
    BOX_FILL_RATIO, BOX_EDGE_MARGIN, BOX_ISOLATION_MARGIN,
    BOX_OCR_UPSCALE, BOX_OCR_CONF_MIN, BOX_MAX_CHARS,
    GDT_AREA_MIN, GDT_AREA_MAX, GDT_ASPECT_MIN, GDT_ASPECT_MAX,
    GDT_FILL_RATIO, GDT_EDGE_MARGIN, GDT_OCR_CONF_MIN, GDT_MAX_TEXT_LEN,
    DEDUP_OVERLAP_THRESHOLD, DEDUP_CENTER_PROXIMITY,
    scale_area, scale_length,
)

def _build_spatial_index(items, cell_size=100):
    """Bucket items into a grid for fast O(1) neighbour lookup."""
    index = {}
    for item in items:
        # Prevent zero-division and cluster correctly
        cx = int(sum(p[0] for p in item['bbox']) / 4 / cell_size)
        cy = int(sum(p[1] for p in item['bbox']) / 4 / cell_size)
        index.setdefault((cx, cy), []).append(item)
    return index

def _get_nearby(index, cx, cy, cell_size=100):
    """Return items in the same and 8 adjacent grid cells."""
    gx, gy = int(cx / cell_size), int(cy / cell_size)
    nearby = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if (gx+dx, gy+dy) in index:
                nearby.extend(index[(gx+dx, gy+dy)])
    return nearby


class BoxCharacterDetector:
    def __init__(self, reader, zoom: float = 2.0):
        """
        Initialize with an existing PaddleOCR reader instance (to avoid
        loading the model twice) and the zoom factor used when rasterising
        the PDF page (default 2.0).

        Args:
            reader : PaddleOCR instance (the same one used in TextExtractor)
            zoom   : Rasterisation zoom (used to scale pixel thresholds)
        """
        from paddleocr import PaddleOCR
        if not isinstance(reader, PaddleOCR):
            raise TypeError(
                f"BoxCharacterDetector expects a PaddleOCR reader, "
                f"got {type(reader).__name__}. "
                f"Pass extractor.reader from your TextExtractor instance."
            )
        self.reader = reader
        self.zoom = zoom
    
    def detect_boxed_characters(self, image, image_gray, existing_items=None, exclusion_items=None):
        """
        Detects isolated rectangular boxes in the drawing and OCRs their contents.
        
        Args:
            image: numpy array (the rasterized page)
            image_gray: pre-computed grayscale image
            existing_items: items already detected by the main OCR pass (to avoid duplicates)
            exclusion_items: items in exclusion zones (to avoid detecting table cells etc.)
            
        Returns:
            new_items: list of dicts with 'bbox', 'text', 'confidence' for newly detected characters
        """
        existing_items  = existing_items  or []
        exclusion_items = exclusion_items or []
        h, w = image.shape[:2]
        
        # 1. Binarize
        gray = image_gray
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Filter for small rectangular contours
        all_rects = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) != 4:
                continue
            
            x, y, bw, bh = cv2.boundingRect(approx)
            area = bw * bh
            aspect = bw / float(bh) if bh > 0 else 0
            contour_area = cv2.contourArea(cnt)
            fill_ratio = contour_area / area if area > 0 else 0
            
            # Datum boxes: small squares — thresholds scale with zoom
            area_min    = scale_area(BOX_AREA_MIN,    self.zoom)
            area_max    = scale_area(BOX_AREA_MAX,    self.zoom)
            edge_margin = scale_length(BOX_EDGE_MARGIN, self.zoom)
            if (area_min < area < area_max and
                BOX_ASPECT_MIN < aspect < BOX_ASPECT_MAX and
                x > edge_margin and y > edge_margin and
                x + bw < w - edge_margin and y + bh < h - edge_margin and
                fill_ratio > BOX_FILL_RATIO):
                all_rects.append((x, y, bw, bh))
        
        # 4. Filter for ISOLATED boxes (not part of GD&T frame arrays)
        isolated = [r for r in all_rects if self._is_isolated(r, all_rects)]
        
        # 5. Remove boxes that overlap with exclusion zones (table, notes)
        if exclusion_items:
            isolated = self._remove_in_exclusion_zones(isolated, exclusion_items, image.shape)
        
        # 6. Remove boxes that overlap with already-detected text items
        if existing_items:
            isolated = self._remove_duplicates(isolated, existing_items)
        
        # 7. OCR each isolated box
        new_items = []
        for (x, y, bw, bh) in isolated:
            text, conf = self._ocr_box(image, x, y, bw, bh)
            
            # Only keep if it looks like a datum/reference marker
            if text and conf > BOX_OCR_CONF_MIN and len(text) <= BOX_MAX_CHARS:
                # Only keep if it looks like a datum/reference marker
                # (single letter, letter+digit, or short number)
                clean = text.strip()
                if re.match(r'^[A-Z]\d?$|^\d{1,2}$', clean, re.IGNORECASE):
                    bbox = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
                    new_items.append({
                        'bbox': bbox,
                        'text': clean,
                        'confidence': conf,
                        'source': 'box_detector'
                    })
        
        return new_items

    def detect_gdt_frames(self, image, image_gray, existing_items=None, exclusion_items=None):
        """
        Detects bordered dimension/datum boxes that EasyOCR misses because they
        are surrounded by a rectangle.
        
        Instead of grouping adjacent boxes (which causes over-merging), this method
        detects each bordered box individually, OCRs it, and accepts the result if it
        looks like a decimal dimension (e.g. 4.88, 0.5) or a short datum reference
        (e.g. F, A B C, 0.5 A B).
        """
        existing_items  = existing_items  or []
        exclusion_items = exclusion_items or []
        import re
        h, w = image.shape[:2]
        gray = image_gray
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Accept any rectangular bordered box of a reasonable size (wider range than
        # detect_boxed_characters which only looks at tiny isolated squares)
        candidate_rects = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            x, y, bw, bh = cv2.boundingRect(approx)
            area = bw * bh
            aspect = bw / float(bh) if bh > 0 else 0
            contour_area = cv2.contourArea(cnt)
            fill_ratio = contour_area / area if area > 0 else 0
            # Allow wider boxes (dimension frames are wide): thresholds scale with zoom
            gdt_area_min    = scale_area(GDT_AREA_MIN,    self.zoom)
            gdt_area_max    = scale_area(GDT_AREA_MAX,    self.zoom)
            gdt_edge_margin = scale_length(GDT_EDGE_MARGIN, self.zoom)
            if (gdt_area_min < area < gdt_area_max and
                GDT_ASPECT_MIN < aspect < GDT_ASPECT_MAX and
                x > gdt_edge_margin and y > gdt_edge_margin and
                x + bw < w - gdt_edge_margin and y + bh < h - gdt_edge_margin and
                fill_ratio > GDT_FILL_RATIO):
                candidate_rects.append((x, y, bw, bh))

        if not candidate_rects:
            return []

        # Skip rects already covered by existing items (centre inside existing bbox)
        # Skip rects already covered by existing items (check for significant area overlap)
        combined_items = existing_items + exclusion_items
        spatial_index = _build_spatial_index(combined_items, cell_size=100)

        def overlaps_existing(rx, ry, rw, rh):
            box_area = rw * rh
            if box_area == 0:
                return False
                
            rcx, rcy = rx + rw / 2, ry + rh / 2
            nearby_items = _get_nearby(spatial_index, rcx, rcy, cell_size=100)
            
            for it in nearby_items:
                ix_min = min(p[0] for p in it['bbox'])
                iy_min = min(p[1] for p in it['bbox'])
                ix_max = max(p[0] for p in it['bbox'])
                iy_max = max(p[1] for p in it['bbox'])
                
                # Intersection
                ox = max(0, min(rx + rw, ix_max) - max(rx, ix_min))
                oy = max(0, min(ry + rh, iy_max) - max(ry, iy_min))
                overlap_area = ox * oy
                
                item_area = (ix_max - ix_min) * (iy_max - iy_min)
                
                # If overlap is > 40% of either box, it's the same item
                if item_area > 0 and overlap_area / min(box_area, item_area) > 0.4:
                    return True
                    
                # Also check if centers are very close
                icx, icy = (ix_min + ix_max) / 2, (iy_min + iy_max) / 2
                prox = scale_length(DEDUP_CENTER_PROXIMITY, self.zoom)
                if abs(rcx - icx) < prox and abs(rcy - icy) < prox:
                    return True
                    
            return False

        # Patterns that are valid annotation content from a bordered box
        decimal_dim   = re.compile(r'\d+[.,]\d+')
        datum_ref     = re.compile(r'^[\d.,\s+⊕⌀Øø\|ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefg]{1,20}$')
        
        new_items = []
        seen_boxes = []  # store (rx, ry, rw, rh) for within-pass dedup
        for (rx, ry, rw, rh) in candidate_rects:
            # Skip if it overlaps an existing item
            if overlaps_existing(rx, ry, rw, rh):
                continue

            # Skip title block / notes region
            rcx = rx + rw / 2
            rcy = ry + rh / 2
            if rcx > w * 0.72 and rcy > h * 0.75:
                continue

            # Dedup within this pass using IoU / area overlap
            is_seen = False
            box_area = rw * rh
            for (sx, sy, sw, sh) in seen_boxes:
                ox = max(0, min(rx + rw, sx + sw) - max(rx, sx))
                oy = max(0, min(ry + rh, sy + sh) - max(ry, sy))
                overlap = ox * oy
                seen_area = sw * sh
                if overlap > 0 and overlap / min(box_area, seen_area) > 0.5:
                    is_seen = True
                    break
            if is_seen:
                continue

            # OCR the individual box
            text, conf = self._ocr_box(image, rx, ry, rw, rh)
            if not text or conf < GDT_OCR_CONF_MIN:
                continue

            clean = text.strip()
            if decimal_dim.search(clean) or datum_ref.match(clean):
                if len(clean) > GDT_MAX_TEXT_LEN:
                    continue
                bbox = [[rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]]
                new_items.append({'bbox': bbox, 'text': clean, 'confidence': conf, 'source': 'gdt_frame_detector'})
                seen_boxes.append((rx, ry, rw, rh))
                print(f"  - GD&T frame detected: '{clean}' @ ({rx},{ry})")

        return new_items

    def _is_isolated(self, rect, all_rects, margin=15):
        """Check if a rectangle has no touching neighbors (including diagonals)."""
        x, y, bw, bh = rect
        cx, cy = x + bw / 2, y + bh / 2
        for ox, oy, obw, obh in all_rects:
            if (ox, oy, obw, obh) == rect:
                continue
            ocx, ocy = ox + obw / 2, oy + obh / 2
            dist = ((cx - ocx)**2 + (cy - ocy)**2) ** 0.5
            # Two boxes are neighbours if their centres are within 2x the average box size
            avg_size = ((bw + bh + obw + obh) / 4)
            if dist < avg_size * 2 + margin:
                return False
        return True
    
    def _remove_in_exclusion_zones(self, rects, exclusion_items, image_shape):
        """Remove boxes that fall inside exclusion zones."""
        h, w = image_shape[:2]
        valid = []
        spatial_index = _build_spatial_index(exclusion_items, cell_size=200)
        
        for (x, y, bw, bh) in rects:
            cx, cy = x + bw // 2, y + bh // 2
            in_exclusion = False
            
            nearby_excl = _get_nearby(spatial_index, cx, cy, cell_size=200)
            for item in nearby_excl:
                bbox = item['bbox']
                ix_min = min(p[0] for p in bbox) - 10
                iy_min = min(p[1] for p in bbox) - 10
                ix_max = max(p[0] for p in bbox) + 10
                iy_max = max(p[1] for p in bbox) + 10
                
                if ix_min <= cx <= ix_max and iy_min <= cy <= iy_max:
                    in_exclusion = True
                    break
            
            if not in_exclusion:
                valid.append((x, y, bw, bh))
        
        return valid
    
    def _remove_duplicates(self, rects, existing_items,
                            overlap_threshold=DEDUP_OVERLAP_THRESHOLD,
                            proximity=None):
        """Remove boxes that significantly overlap with or are very close to already-detected text items."""
        if proximity is None:
            proximity = scale_length(DEDUP_CENTER_PROXIMITY, self.zoom)
        valid = []
        spatial_index = _build_spatial_index(existing_items, cell_size=100)
        
        for (x, y, bw, bh) in rects:
            is_duplicate = False
            box_cx = x + bw / 2
            box_cy = y + bh / 2
            
            nearby_items = _get_nearby(spatial_index, box_cx, box_cy, cell_size=100)
            for item in nearby_items:
                bbox = item['bbox']
                ix_min = min(p[0] for p in bbox)
                iy_min = min(p[1] for p in bbox)
                ix_max = max(p[0] for p in bbox)
                iy_max = max(p[1] for p in bbox)
                
                # Check 1: Bbox overlap
                ox = max(0, min(x + bw, ix_max) - max(x, ix_min))
                oy = max(0, min(y + bh, iy_max) - max(y, iy_min))
                overlap_area = ox * oy
                box_area = bw * bh
                
                if box_area > 0 and overlap_area / box_area > overlap_threshold:
                    is_duplicate = True
                    break
                
                # Check 2: Center proximity — if centers are within `proximity` px,
                # it's likely the same text detected by both EasyOCR and the box detector
                item_cx = sum(p[0] for p in bbox) / len(bbox)
                item_cy = sum(p[1] for p in bbox) / len(bbox)
                dist = ((box_cx - item_cx) ** 2 + (box_cy - item_cy) ** 2) ** 0.5
                
                if dist < proximity:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                valid.append((x, y, bw, bh))
        
        return valid
    
    def _ocr_box(self, image, x, y, bw, bh):
        """Crop and OCR a single box region."""
        pad = 3
        crop = image[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]
        
        # Upscale for better OCR accuracy
        upscale = BOX_OCR_UPSCALE
        crop_big = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        
        try:
            results = self.reader.ocr(crop_big)
            if not results or not results[0]:
                return '', 0.0
            text = ' '.join([r[1][0] for r in results[0]]).strip()
            conf = max([r[1][1] for r in results[0]])
            return text, conf
        except (TypeError, IndexError, AttributeError) as e:
            print(f"  [_ocr_box] OCR format error at ({x},{y}): {e}")
            return '', 0.0
