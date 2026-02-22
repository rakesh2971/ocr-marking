import cv2
import numpy as np
from PIL import Image
from processor import DocumentProcessor
from extractor import TextExtractor

def dist(cx, cy, bx, by, bw, bh):
    """Squared distance from point to nearest point on rectangle."""
    dx = max(0, bx - cx, cx - (bx + bw))
    dy = max(0, by - cy, cy - (by + bh))
    return dx*dx + dy*dy

def run():
    print("Loading image...")
    proc = DocumentProcessor()
    # Ensure zoom=2.0 as in main pipeline
    images = proc.pdf_to_images("LCA.pdf", zoom=2.0) 
    img = images[0]

    # Extract text items so we can erase them and re-assign them
    print("Extracting text...")
    extractor = TextExtractor()
    raw_text = extractor.extract_text_custom(img, x_threshold=40)

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 1. Start with the binary image of all black ink
    draw_ink = binary.copy()

    # 2. Erase long lines (page borders, title block layout lines)
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 5, 1))
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 5))
    draw_ink[cv2.morphologyEx(draw_ink, cv2.MORPH_OPEN, h_kern) > 0] = 0
    draw_ink[cv2.morphologyEx(draw_ink, cv2.MORPH_OPEN, v_kern) > 0] = 0

    # 3. Erase the notes section (rightmost 27%) and the title block region
    # Erasing the notes explicitly here guarantees no clusters form in that area
    cv2.rectangle(draw_ink, (int(w * 0.73), 0), (w, h), 0, -1)
    # The title block extends further left, erase it too
    cv2.rectangle(draw_ink, (int(w * 0.65), int(h * 0.8)), (w, h), 0, -1)

    # 4. Erase all text boxes to prevent text from bridging distinct drawing views
    for item in raw_text:
        bbox = item['bbox']
        x_min = int(min(p[0] for p in bbox))
        x_max = int(max(p[0] for p in bbox))
        y_min = int(min(p[1] for p in bbox))
        y_max = int(max(p[1] for p in bbox))
        
        # Expand slightly to erase all text ink completely
        pad = 5
        cv2.rectangle(draw_ink, (max(0, x_min-pad), max(0, y_min-pad)), (min(w, x_max+pad), min(h, y_max+pad)), 0, -1)

    # Now we have primarily the actual drawing parts ink.
    # 5. Dilate to group disconnected lines of a single part into blobs using k90
    print("\n--- Running k90 Clustering ---")
    k_size = 90
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(draw_ink, kernel, iterations=1)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    # Filter blobs by minimum area
    min_area = (h * w) * 0.0015 # 0.15% of page area
    
    initial_boxes = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            initial_boxes.append([x, y, bw, bh])
            
    # Dissolve smaller clusters that are completely inside a larger cluster
    part_boxes = []
    for i, box_inner in enumerate(initial_boxes):
        ix, iy, iw, ih = box_inner
        is_inside = False
        for j, box_outer in enumerate(initial_boxes):
            if i == j:
                continue
            ox, oy, ow, oh = box_outer
            
            # Check if box_inner is 100% inside box_outer
            if (ix >= ox and iy >= oy and 
                (ix + iw) <= (ox + ow) and 
                (iy + ih) <= (oy + oh)):
                is_inside = True
                break
                
        if not is_inside:
            part_boxes.append(box_inner)
            
    # Sort part_boxes top-to-bottom, left-to-right using row binning
    # Create roughly 600px rows
    ROW_HEIGHT = 600
    if h > 1200:
        num_rows = max(4, min(12, h // 400))
        ROW_HEIGHT = max(400, h // num_rows)
        
    part_boxes.sort(key=lambda b: (b[1] // ROW_HEIGHT, b[0]))
            
    print(f"Found {len(part_boxes)} drawing parts (after dissolving inner clusters).")
    
    # 6. Assign annotations to nearest drawing part
    view_annotations = {i: [] for i in range(len(part_boxes))}
    
    assigned_count = 0
    for item in raw_text:
        bbox = item['bbox']
        cx = sum(p[0] for p in bbox) / 4
        cy = sum(p[1] for p in bbox) / 4
        
        # Skip annotations in the notes section (rightmost 27%) entirely
        if cx > w * 0.73:
            continue
            
        # Skip title block annotations roughly
        if cx > w * 0.65 and cy > h * 0.8:
            continue
            
        best_idx = None
        best_d = float('inf')
        
        # Find nearest view
        for idx, (bx, by, bw, bh) in enumerate(part_boxes):
            d = dist(cx, cy, bx, by, bw, bh)
            if d < best_d:
                best_d = d
                best_idx = idx
        
        if best_idx is not None:
            view_annotations[best_idx].append(item)
            assigned_count += 1
            
    print(f"Assigned {assigned_count} annotations.")
    
    # 7. Compute final cluster bounding boxes (part box + its annotations)
    out_img = img.copy()
    
    for idx, (px, py, pw, ph) in enumerate(part_boxes):
        min_x, min_y = px, py
        max_x, max_y = px + pw, py + ph
        
        # Expand with assigned annotations
        for item in view_annotations[idx]:
            bbox = item['bbox']
            for p in bbox:
                min_x = min(min_x, int(p[0]))
                max_x = max(max_x, int(p[0]))
                min_y = min(min_y, int(p[1]))
                max_y = max(max_y, int(p[1]))
                
        # Draw Blue Bounding Box for the cluster
        cv2.rectangle(out_img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 4)
        # Add a Label
        cv2.putText(out_img, f"Cluster {idx+1}", (min_x, max(0, min_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Resize for saving so it's not gigantic
    out_resize = cv2.resize(out_img, (w//4, h//4))
    out_file = f"debug_clusters_k90.png"
    cv2.imwrite(out_file, out_resize)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    run()
