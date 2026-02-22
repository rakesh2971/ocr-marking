"""Diagnostic: try contour-based rectangle detection for view boxes."""
import cv2
import numpy as np
from processor import DocumentProcessor

proc = DocumentProcessor()
images = proc.pdf_to_images("LCA.pdf", zoom=2.0)
image = images[0]
h_img, w_img = image.shape[:2]
print(f"Image size: {w_img} x {h_img}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# --- Approach: find rectangular contours directly ---
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

page_area = h_img * w_img
min_area = page_area * 0.03   # at least 3% of page
max_area = page_area * 0.90   # skip full-page border

rects = []
for i, cnt in enumerate(contours):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) < 4 or len(approx) > 6:  # roughly rectangular
        continue

    x, y, bw, bh = cv2.boundingRect(cnt)
    area = bw * bh

    if area < min_area or area > max_area:
        continue

    # Check rectangularity: contour area vs bounding rect area
    cnt_area = cv2.contourArea(cnt)
    rect_ratio = cnt_area / area if area > 0 else 0

    if rect_ratio > 0.85:  # must be mostly rectangular
        parent = hierarchy[0][i][3] if hierarchy is not None else -1
        rects.append((x, y, bw, bh, area, rect_ratio, len(approx), parent))

print(f"\nFound {len(rects)} rectangular regions (3-90% of page):")
for i, (x, y, bw, bh, area, ratio, npts, parent) in enumerate(rects):
    pct = area / page_area * 100
    print(f"  Rect {i+1}: x={x:<5} y={y:<5} w={bw:<5} h={bh:<5} area={pct:5.1f}% rect_ratio={ratio:.2f} pts={npts} parent={parent}")

# --- Also try: use the line-structure approach but look at what lines exist ---
print("\n\n--- Line analysis ---")
h_lengths = [50, 100, 150, 200, 250]
for hl in h_lengths:
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (hl, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kern)
    count = cv2.connectedComponentsWithStats(h_lines)[0] - 1
    print(f"  H-lines >= {hl}px: {count} segments")

v_lengths = [50, 100, 150, 200, 250]
for vl in v_lengths:
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vl))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kern)
    count = cv2.connectedComponentsWithStats(v_lines)[0] - 1
    print(f"  V-lines >= {vl}px: {count} segments")
