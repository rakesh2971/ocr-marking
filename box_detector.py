"""
box_detector.py - Detects isolated single-character boxes (datum/reference markers)
in engineering drawings using OpenCV contour detection + targeted OCR.

These are standalone small rectangles containing characters like X, E, Z, Z1, B
that EasyOCR's full-page scan often misses.
"""
import cv2
import numpy as np
import re


class BoxCharacterDetector:
    def __init__(self, reader):
        """
        Initialize with an existing EasyOCR reader instance (to avoid loading model twice).
        """
        self.reader = reader
    
    def detect_boxed_characters(self, image, existing_items=[], exclusion_items=[]):
        """
        Detects isolated rectangular boxes in the drawing and OCRs their contents.
        
        Args:
            image: numpy array (the rasterized page)
            existing_items: items already detected by the main OCR pass (to avoid duplicates)
            exclusion_items: items in exclusion zones (to avoid detecting table cells etc.)
            
        Returns:
            new_items: list of dicts with 'bbox', 'text', 'confidence' for newly detected characters
        """
        h, w = image.shape[:2]
        
        # 1. Binarize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
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
            
            # Datum boxes: roughly 30-50px squares at zoom=2.0, high fill ratio
            if (800 < area < 4500 and 
                0.6 < aspect < 2.8 and 
                x > 50 and y > 50 and x + bw < w - 50 and y + bh < h - 50 and
                fill_ratio > 0.85):
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
            
            # Filter: must have valid text and reasonable confidence
            if text and conf > 0.1 and len(text) <= 4:
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

    def detect_gdt_frames(self, image, existing_items=[], exclusion_items=[]):
        """
        Detects bordered dimension/datum boxes that EasyOCR misses because they
        are surrounded by a rectangle.
        
        Instead of grouping adjacent boxes (which causes over-merging), this method
        detects each bordered box individually, OCRs it, and accepts the result if it
        looks like a decimal dimension (e.g. 4.88, 0.5) or a short datum reference
        (e.g. F, A B C, 0.5 A B).
        """
        import re
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
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
            # Allow wider boxes (dimension frames are wide): up to 20000 px area
            # and aspect ratios up to 10 (wide dimension boxes)
            if (700 < area < 20000 and
                0.4 < aspect < 10.0 and
                x > 50 and y > 50 and x + bw < w - 50 and y + bh < h - 50 and
                fill_ratio > 0.75):
                candidate_rects.append((x, y, bw, bh))

        if not candidate_rects:
            return []

        # Skip rects already covered by existing items (centre inside existing bbox)
        def overlaps_existing(rx, ry, rw, rh):
            rcx, rcy = rx + rw / 2, ry + rh / 2
            for it in existing_items:
                ix_min = min(p[0] for p in it['bbox'])
                iy_min = min(p[1] for p in it['bbox'])
                ix_max = max(p[0] for p in it['bbox'])
                iy_max = max(p[1] for p in it['bbox'])
                if ix_min <= rcx <= ix_max and iy_min <= rcy <= iy_max:
                    return True
            return False

        # Patterns that are valid annotation content from a bordered box
        decimal_dim   = re.compile(r'\d+[.,]\d+')
        datum_ref     = re.compile(r'^[\d.,\s+⊕⌀Øø\|ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefg]{1,20}$')
        
        new_items = []
        seen_centers = []  # dedup within this pass
        for (rx, ry, rw, rh) in candidate_rects:
            # Skip if it overlaps an existing item
            if overlaps_existing(rx, ry, rw, rh):
                continue

            # Skip title block / notes region
            rcx = rx + rw / 2
            rcy = ry + rh / 2
            if rcx > w * 0.72 and rcy > h * 0.75:
                continue

            # Dedup within this pass
            if any(abs(rcx - sc[0]) < 50 and abs(rcy - sc[1]) < 50 for sc in seen_centers):
                continue

            # OCR the individual box
            text, conf = self._ocr_box(image, rx, ry, rw, rh)
            if not text or conf < 0.15:
                continue

            clean = text.strip()
            # Accept only if it contains a decimal dimension OR looks like a datum ref
            if decimal_dim.search(clean) or datum_ref.match(clean):
                # Extra sanity: reject very long strings (likely notes sneaking through)
                if len(clean) > 25:
                    continue
                bbox = [[rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]]
                new_items.append({'bbox': bbox, 'text': clean, 'confidence': conf, 'source': 'gdt_frame_detector'})
                seen_centers.append((rcx, rcy))
                print(f"  - GD&T frame detected: '{clean}' @ ({rx},{ry})")

        return new_items

    def _is_isolated(self, rect, all_rects, margin=5):
        """Check if a rectangle has no touching neighbors on the same row."""
        x, y, bw, bh = rect
        neighbors = 0
        for ox, oy, obw, obh in all_rects:
            if (ox, oy, obw, obh) == rect:
                continue
            # Check vertical alignment
            y_overlap = max(0, min(y + bh, oy + obh) - max(y, oy))
            if y_overlap < bh * 0.5:
                continue
            # Check horizontal touching
            h_gap = min(abs(x + bw - ox), abs(ox + obw - x))
            if h_gap < margin:
                neighbors += 1
        return neighbors == 0
    
    def _remove_in_exclusion_zones(self, rects, exclusion_items, image_shape):
        """Remove boxes that fall inside exclusion zones."""
        h, w = image_shape[:2]
        valid = []
        
        for (x, y, bw, bh) in rects:
            cx, cy = x + bw // 2, y + bh // 2
            in_exclusion = False
            
            for item in exclusion_items:
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
    
    def _remove_duplicates(self, rects, existing_items, overlap_threshold=0.5, proximity=40):
        """Remove boxes that significantly overlap with or are very close to already-detected text items."""
        valid = []
        
        for (x, y, bw, bh) in rects:
            is_duplicate = False
            box_cx = x + bw / 2
            box_cy = y + bh / 2
            
            for item in existing_items:
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
        crop_big = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        
        results = self.reader.readtext(crop_big)
        
        if results:
            text = ' '.join([r[1] for r in results]).strip()
            conf = max([r[2] for r in results])
            return text, conf
        
        return '', 0.0
