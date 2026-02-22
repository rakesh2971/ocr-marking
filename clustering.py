
import cv2
import numpy as np


def _point_in_rect(cx, cy, x, y, w, h):
    """True if (cx, cy) is inside rectangle (x, y, w, h)."""
    return x <= cx <= x + w and y <= cy <= y + h


def _dist_point_to_rect(cx, cy, x, y, w, h):
    """Squared distance from point (cx, cy) to nearest point on rect. 0 if inside."""
    dx = max(0, x - cx, cx - (x + w))
    dy = max(0, y - cy, cy - (y + h))
    return dx * dx + dy * dy


class MorphologicalClusterer:
    def __init__(self):
        pass

    def _get_view_regions(self, image, items, exclusion_items):
        """
        Detects major drawing parts/views by heavy morphological dilation.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        draw_ink = binary.copy()
        
        h, w = draw_ink.shape

        # Erase long straight lines that might connect parts
        h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 5, 1))
        v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 5))
        draw_ink[cv2.morphologyEx(draw_ink, cv2.MORPH_OPEN, h_kern) > 0] = 0
        draw_ink[cv2.morphologyEx(draw_ink, cv2.MORPH_OPEN, v_kern) > 0] = 0

        # Erase the notes section (rightmost 27%) and the title block region
        cv2.rectangle(draw_ink, (int(w * 0.73), 0), (w, h), 0, -1)
        cv2.rectangle(draw_ink, (int(w * 0.65), int(h * 0.8)), (w, h), 0, -1)

        # Erase all text boxes to prevent text from bridging distinct drawing views
        for item in items:
            bbox = item['bbox']
            x_min = int(min(p[0] for p in bbox))
            x_max = int(max(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))
            y_max = int(max(p[1] for p in bbox))
            pad = 5
            cv2.rectangle(draw_ink, (max(0, x_min-pad), max(0, y_min-pad)), (min(w, x_max+pad), min(h, y_max+pad)), 0, -1)

        # Heavy dilation to merge parts and their annotations into single blobs
        k_size = 90
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(draw_ink, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        page_area = h * w
        min_area = page_area * 0.0015  # 0.15% of page minimum size

        initial_boxes = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                bx = stats[i, cv2.CC_STAT_LEFT]
                by = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                initial_boxes.append((bx, by, bw, bh))

        # Dissolve smaller clusters that are completely inside a larger cluster
        part_boxes = []
        for i, box_inner in enumerate(initial_boxes):
            ix, iy, iw, ih = box_inner
            is_inside = False
            for j, box_outer in enumerate(initial_boxes):
                if i == j:
                    continue
                ox, oy, ow, oh = box_outer
                if (ix >= ox and iy >= oy and 
                    (ix + iw) <= (ox + ow) and 
                    (iy + ih) <= (oy + oh)):
                    is_inside = True
                    break
            if not is_inside:
                part_boxes.append(box_inner)

        # Sort part_boxes top-to-bottom, left-to-right using row binning
        ROW_HEIGHT = 600
        if h > 1200:
            num_rows = max(4, min(12, h // 400))
            ROW_HEIGHT = max(400, h // num_rows)
            
        part_boxes.sort(key=lambda b: (b[1] // ROW_HEIGHT, b[0]))
        return part_boxes

    def get_clusters(self, image, items, exclusion_items=[]):
        """
        Groups annotations by drawing view regions (layout), not by proximity.

        - Detects major drawing views (Front, Side, Sections, Details) as layout regions.
        - Assigns each annotation to its containing view, or nearest view if none contains.
        - Sorts views in reading order (Top → Bottom, Left → Right).
        - Sorts annotations within each view in reading order.
        - Returns (cluster_info, labeled_img) for sequential view-by-view numbering.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape

        view_rects = self._get_view_regions(image, items, exclusion_items)
        if not view_rects:
            view_rects = [(0, 0, w, h)]

        view_to_items = {i: [] for i in range(len(view_rects))}

        for item in items:
            bbox = item['bbox']
            cx = sum(p[0] for p in bbox) / 4
            cy = sum(p[1] for p in bbox) / 4

            # Skip annotations in the notes section (rightmost 27%) entirely
            if cx > w * 0.73:
                continue
                
            # Skip title block annotations roughly
            if cx > w * 0.65 and cy > h * 0.8:
                continue

            best_view = None
            best_dist_sq = float('inf')

            for vi, (rx, ry, rw, rh) in enumerate(view_rects):
                if _point_in_rect(cx, cy, rx, ry, rw, rh):
                    best_view = vi
                    best_dist_sq = 0
                    break
                d2 = _dist_point_to_rect(cx, cy, rx, ry, rw, rh)
                if d2 < best_dist_sq:
                    best_dist_sq = d2
                    best_view = vi

            if best_view is not None:
                view_to_items[best_view].append(item)

        # ── Row binning: 600px nominal rows (or ~8 rows if image taller) ──
        # Each cluster is assigned to the row that contains its top edge (ry).
        # Use 600px as requested, but ensure we get multiple rows on tall images.
        ROW_HEIGHT = 600
        if h > 1200:
            # Scale so we get ~8 rows on tall pages (A3 at zoom 2 is ~5000px tall)
            num_rows = max(4, min(12, h // 400))
            ROW_HEIGHT = max(400, h // num_rows)
        row_to_views = {}  # row_id -> list of (vi, (rx, ry, rw, rh))
        for vi, (rx, ry, rw, rh) in enumerate(view_rects):
            if not view_to_items[vi]:
                continue
            row_id = int(ry) // ROW_HEIGHT
            row_to_views.setdefault(row_id, []).append((vi, (rx, ry, rw, rh)))

        # Within each row, sort clusters left→right (by rx)
        for row_id in row_to_views:
            row_to_views[row_id].sort(key=lambda t: t[1][0])

        # Build cluster_info: process row 0, then row 1, ...; within row process clusters in order
        cluster_info = []
        for row_id in sorted(row_to_views.keys()):
            for vi, (rx, ry, rw, rh) in row_to_views[row_id]:
                cluster_items = view_to_items[vi]
                if not cluster_items:
                    continue

                # Reading order within cluster: top→bottom, left→right
                cluster_items.sort(key=lambda it: (
                    min(p[1] for p in it['bbox']),
                    min(p[0] for p in it['bbox'])
                ))

                min_x = min(min(p[0] for p in it['bbox']) for it in cluster_items)
                centroid_y = int(np.mean([
                    sum(p[1] for p in it['bbox']) / 4 for it in cluster_items
                ]))
                cluster_info.append((vi, min_x, centroid_y, cluster_items))

        # Debug: confirm row binning is applied
        n_rows = len(row_to_views)
        if n_rows > 0:
            row_summary = ", ".join(f"row{r}:{len(row_to_views[r])} clusters" for r in sorted(row_to_views.keys()))
            print(f"  Row binning: height={ROW_HEIGHT}px, {n_rows} rows ({row_summary})")

        labeled_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for vi, (rx, ry, rw, rh) in enumerate(view_rects):
            color = (
                (37 * (vi + 1)) % 256,
                (97 * (vi + 1) + 50) % 256,
                (157 * (vi + 1) + 100) % 256,
            )
            cv2.rectangle(labeled_img, (rx, ry), (rx + rw, ry + rh), color, 2)

        return cluster_info, labeled_img
