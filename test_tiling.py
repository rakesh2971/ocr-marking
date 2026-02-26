import cv2; import numpy as np; from processor import DocumentProcessor
p=DocumentProcessor(); img=p.pdf_to_images('LCA_1.pdf', zoom=2)[0]
gray=cv2.medianBlur(img if len(img.shape)==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
h, w = gray.shape
T = 2000
overlap = 100
circles=[]
for y in range(0, h, T):
    for x in range(0, w, T):
        y_max = min(h, y+T+overlap)
        x_max = min(w, x+T+overlap)
        crop=gray[y:y_max, x:x_max]
        c=cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=65)
        if c is not None:
            for cx, cy, r in c[0]: 
                circles.append((cx+x, cy+y, r))

# Dedup circles that appear in overlapping regions
deduped = []
for cx, cy, cr in circles:
    dup = False
    for dx, dy, dr in deduped:
        if ((cx-dx)**2 + (cy-dy)**2)**0.5 < 15:
            dup = True
            break
    if not dup:
        deduped.append((cx, cy, cr))                

dists=[((cx-2038)**2+(cy-441)**2)**0.5 for cx,cy,_ in deduped]
min_dist=min(dists) if dists else 9999
print('Min dist:', min_dist, 'Circles found:', len(deduped))
