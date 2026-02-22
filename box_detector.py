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
        Detects GD&T feature control frames (multi-compartment horizontal groups of boxes)
        that are NOT isolated (so they're filtered out by detect_boxed_characters).
        Groups adjacent boxes in the same row and OCRs the entire group as one item.
        Also catches datum boxes (single boxed letter) that might have been missed.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Collect all smallish rectangles (relaxed area range)
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
            if (500 < area < 8000 and
                0.3 < aspect < 5.0 and
                x > 50 and y > 50 and x + bw < w - 50 and y + bh < h - 50 and
                fill_ratio > 0.75):
                all_rects.append((x, y, bw, bh))

        if not all_rects:
            return []

        # Skip isolated boxes (those are handled by detect_boxed_characters)
        non_isolated = [r for r in all_rects if not self._is_isolated(r, all_rects)]

        # Group non-isolated boxes that are on the same row and touching/close
        groups = []
        used = [False] * len(non_isolated)
        for i, (x, y, bw, bh) in enumerate(non_isolated):
            if used[i]:
                continue
            group = [(x, y, bw, bh)]
            used[i] = True
            for j, (ox, oy, obw, obh) in enumerate(non_isolated):
                if used[j]:
                    continue
                # Same row (centres within bh of each other)
                cy_i, cy_j = y + bh / 2, oy + obh / 2
                if abs(cy_i - cy_j) > max(bh, obh) * 0.6:
                    continue
                # Horizontally close
                gap = abs(x + bw - ox) if x < ox else abs(ox + obw - x)
                if gap < 20:
                    group.append((ox, oy, obw, obh))
                    used[j] = True
            if len(group) >= 2:  # Only keep multi-compartment groups
                groups.append(group)

        new_items = []
        for group in groups:
            gx_min = min(r[0] for r in group)
            gy_min = min(r[1] for r in group)
            gx_max = max(r[0] + r[2] for r in group)
            gy_max = max(r[1] + r[3] for r in group)

            # Check it doesn't overlap with existing items
            gcx = (gx_min + gx_max) / 2
            gcy = (gy_min + gy_max) / 2
            already = any(
                abs(sum(p[0] for p in it['bbox']) / 4 - gcx) < 80 and
                abs(sum(p[1] for p in it['bbox']) / 4 - gcy) < 80
                for it in existing_items
            )
            if already:
                continue

            # Skip if in title block / notes area
            if gcx > w * 0.72 and gcy > h * 0.75:
                continue

            # OCR the whole group region
            pad = 5
            crop = image[max(0, gy_min - pad):gy_max + pad, max(0, gx_min - pad):gx_max + pad]
            crop_big = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            results = self.reader.readtext(crop_big, paragraph=False)
            if not results:
                continue
            text = ' '.join([r[1] for r in results]).strip()
            conf = max([r[2] for r in results])

            if not text or conf < 0.1:
                continue

            bbox = [[gx_min, gy_min], [gx_max, gy_min], [gx_max, gy_max], [gx_min, gy_max]]
            new_items.append({'bbox': bbox, 'text': text, 'confidence': conf, 'source': 'gdt_frame_detector'})
            print(f"  - GD&T frame detected: '{text}' @ ({gx_min},{gy_min})")

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
