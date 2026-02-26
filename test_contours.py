import cv2
import numpy as np

def run_test():
    img = cv2.imread('debug_y_circle.png', 0)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    print(f"Total contours: {len(contours)}")
    for c in contours:
        area = cv2.contourArea(c)
        if area > 300 and area < 20000:
            peri = cv2.arcLength(c, True)
            if peri == 0: continue
            circularity = 4 * np.pi * (area / (peri * peri))
            x, y, w, h = cv2.boundingRect(c)
            aspect = float(w) / h
            if 0.6 < aspect < 1.4 and circularity > 0.4: # relax circularity for ellipses and leader-line connected circles
                r = max(w, h) / 2
                if r > 20: 
                    circles.append((x + w/2, y + h/2, r))
                    print(f"Circle at {int(x+w/2)}, {int(y+h/2)} r={r:.1f} circ={circularity:.2f} aspect={aspect:.2f} area={area:.0f}")

    print('Found contour circles:', len(circles))

if __name__ == "__main__":
    run_test()
