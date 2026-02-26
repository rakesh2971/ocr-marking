import cv2
import numpy as np
from processor import DocumentProcessor
from filter import AnnotationFilter

def run_debug():
    p = DocumentProcessor()
    img = p.pdf_to_images('LCA_1.pdf', zoom=2.0)[0]
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    circles_found = []
    for c in contours:
        area = cv2.contourArea(c)
        if 300 < area < 20000:
            peri = cv2.arcLength(c, True)
            if peri == 0: continue
            
            circularity = 4 * np.pi * (area / (peri * peri))
            x, y, w, h = cv2.boundingRect(c)
            if h == 0: continue
            aspect = float(w) / h
            
            if 0.6 < aspect < 1.4 and circularity > 0.4:
                r = max(w, h) / 2.0
                if r > 10:
                    circles_found.append((x + w/2.0, y + h/2.0, r, area, circularity, aspect))
                    
    print(f"Found {len(circles_found)} contour circles.")
    # Show the 10 biggest and 10 smallest
    circles_found.sort(key=lambda x: x[2], reverse=True)
    print("Top 15 biggest circles:")
    for cx, cy, r, a, circ, asp in circles_found[:15]:
        print(f"  r={r:.1f} at ({cx:.1f}, {cy:.1f}), area={a:.0f}, circ={circ:.2f}, aspect={asp:.2f}")

if __name__ == "__main__":
    run_debug()
