from filter import AnnotationFilter
from vector_extractor import VectorExtractor
from processor import DocumentProcessor
from extractor import TextExtractor

def run_debug():
    p = DocumentProcessor()
    fitz_doc = p.open_doc('LCA_1.pdf')
    vec = VectorExtractor()
    extractor = TextExtractor()
    raw_items = vec.extract_page_items(fitz_doc[0], zoom=2.0)
    raw_items = vec.filter_items(raw_items, extractor)
    print(f"Total raw vector items: {len(raw_items)}")

    img = p.pdf_to_images('LCA_1.pdf', zoom=2.0)[0]
    af = AnnotationFilter()
    circles = af.detect_black_circles(img)
    
    # Track sizes
    valid_items_1, excluded_datums = af.filter_by_circles(raw_items, circles)
    print(f"Datums dropped: {len(excluded_datums)}")
    
    valid_items_2, excluded_notes = af.filter_notes_section(valid_items_1)
    print(f"Notes dropped: {len(excluded_notes)}")
    
    rescued_notes, excluded_notes = af.rescue_from_notes_filter(excluded_notes, raw_items)
    if rescued_notes: valid_items_2 += rescued_notes
    print(f"Notes rescued: {len(rescued_notes)}")
    
    valid_items_3, excluded_table = af.filter_bottom_right_table(valid_items_2, img.shape)
    print(f"Table dropped: {len(excluded_table)}")
    
    valid_items_4, excluded_views = af.filter_view_labels(valid_items_3)
    print(f"Views dropped: {len(excluded_views)}")
    
    valid_items_5, excluded_top_left = af.filter_top_left_numbers(valid_items_4, img.shape)
    print(f"Top-Left dropped: {len(excluded_top_left)}")
    
    print("\n--- Items dropped by Notes filter ---")
    for item in excluded_notes:
        d = item.get('text', '')
        if d.strip() in ['Y', '-800', '-600', '-400', 'X', '11.0', '-200', '0']: # sample elements that might be missing
            print(f"NOTE DROP: {d}")
    print("Total notes dropped:", len(excluded_notes))
            
    print("\n--- Items dropped by Table filter ---")
    dropped_table_texts = [i.get('text', '') for i in excluded_table]
    print(f"Sample of {len(dropped_table_texts)} items dropped by Table:")
    print(dropped_table_texts[:30])
    print("Found right side datums in table drop?", any(x in ['-800', '-600', '-400', '-200'] for x in dropped_table_texts))
    
    print("\n--- Items dropped by Top-Left filter ---")
    dropped_top = [i.get('text', '') for i in excluded_top_left]
    print(f"Sample dropped by Top-Left:")
    print(dropped_top[:10])

if __name__ == "__main__":
    try:
        run_debug()
    except Exception as e:
        import traceback
        traceback.print_exc()
