import fitz
from processor import DocumentProcessor
from filter import AnnotationFilter
from vector_extractor import VectorExtractor

def run_debug():
    proc = DocumentProcessor()
    fitz_doc = proc.open_doc("LCA_1.pdf")
    zoom = 2.0
    images = proc.pdf_to_images("LCA_1.pdf", zoom=zoom)
    img = images[0]

    annot_filter = AnnotationFilter()
    circles = annot_filter.detect_black_circles(img)
    print(f"Found {len(circles)} circles.")

    vec_ext = VectorExtractor()
    raw_items = vec_ext.extract_page_items(fitz_doc[0], zoom=zoom)
    print(f"Extracted {len(raw_items)} raw vector items.")

    # Find the "Y" item and "-800" or "-600"
    targets = []
    for item in raw_items:
        text = item['text'].strip()
        if text in ["Y", "-800", "-600", "-400"]:
            targets.append(item)

    print(f"Found {len(targets)} targets matching Y, -800, -600, -400")
    
    # Check them against circles
    for target in targets:
        bbox = target['bbox']
        text = target['text']
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        cx = sum(x_coords) / 4
        cy = sum(y_coords) / 4

        # Find closest circle
        closest_dist = 999999
        closest_c = None
        for (ccx, ccy, cr) in circles:
            dist = ((cx - ccx)**2 + (cy - ccy)**2)**0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_c = (ccx, ccy, cr)

        if closest_c:
            ccx, ccy, cr = closest_c
            print(f"Text '{text}' at ({cx:.1f}, {cy:.1f}). Closest circle at ({ccx}, {ccy}) r={cr}. Dist={closest_dist:.1f}")
            inside = annot_filter.is_inside_circle(bbox, circles)
            print(f"  -> is_inside_circle: {inside}")

if __name__ == "__main__":
    run_debug()
