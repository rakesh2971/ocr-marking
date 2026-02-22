import cv2
import numpy as np
from processor import DocumentProcessor

def test_grid_clustering(image_path):
    proc = DocumentProcessor()
    images = proc.pdf_to_images(image_path, zoom=2.0)
    img = images[0]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    h, w = binary.shape
    
    # 1. Extract Horizontal Lines
    # Use shorter threshold to grab more line segments, even broken ones,
    # then dilate horizontally to connect them.
    h_len = 150 
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kern)
    # Thicken horizontally to close broken segments, then vertically to make thick boundaries
    h_lines = cv2.dilate(h_lines, np.ones((1, 50), np.uint8), iterations=1)
    h_lines = cv2.dilate(h_lines, np.ones((5, 1), np.uint8), iterations=1)
    
    # 2. Extract Vertical Lines
    v_len = 150
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kern)
    # Thicken vertically to close broken segments, then horizontally to make thick boundaries
    v_lines = cv2.dilate(v_lines, np.ones((50, 1), np.uint8), iterations=1)
    v_lines = cv2.dilate(v_lines, np.ones((1, 5), np.uint8), iterations=1)
    
    # 3. Combine into Grid
    grid = cv2.bitwise_or(h_lines, v_lines)
    
    # Dilate the grid a bit more to ensure intersections are solid
    grid = cv2.dilate(grid, np.ones((11, 11), np.uint8), iterations=1)
    
    # 4. Find enclosed regions
    grid_inv = cv2.bitwise_not(grid)
    
    # Also add a border to the grid so the outer edge is closed
    cv2.rectangle(grid_inv, (0, 0), (w-1, h-1), 0, 5)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grid_inv, connectivity=4)
    
    out_img = img.copy()
    
    page_area = h * w
    min_area = page_area * 0.015 # At least 1.5% of page area for a drawing view
    max_area = page_area * 0.95  # Ignore the "entire page" background
    
    count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Skip title block region heavily
            # If the box is small and mostly in the bottom right, skip it
            cx, cy = x + bw/2, y + bh/2
            if cx > w * 0.7 and cy > h * 0.8 and bw < w * 0.35:
                continue
                
            cv2.rectangle(out_img, (x, y), (x+bw, y+bh), (255, 0, 0), 4)
            cv2.putText(out_img, f"Grid {count+1}", (x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            count += 1
            
    print(f"Detected {count} Grid layout cells.")
    
    out_resize = cv2.resize(out_img, (w//4, h//4))
    cv2.imwrite("debug_grid_clusters.png", out_resize)
    print("Saved debug_grid_clusters.png")

if __name__ == "__main__":
    test_grid_clustering("LCA.pdf")
