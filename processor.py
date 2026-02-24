
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import io
import os

class DocumentProcessor:
    def __init__(self):
        pass

    def open_doc(self, pdf_path):
        """
        Return the raw fitz.Document for the given PDF.
        Used by the vector path so the document is opened once and reused
        across all pages without re-opening the file per page.
        """
        return fitz.open(pdf_path)

    def pdf_to_images(self, pdf_path, zoom=2.0):
        """
        Converts a PDF to a list of images (numpy arrays for OpenCV).
        Proactively handles high resolution for better OCR.
        """
        doc = fitz.open(pdf_path)
        images = []
        
        # Matrix for zooming (2.0 = 200% zoom)
        mat = fitz.Matrix(zoom, zoom)
        
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            # Convert to numpy array (H, W, 3)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # If the image has an alpha channel, remove it
            if pix.n == 4:
                img_data = img_data[:, :, :3]
            
            # Use RGB
            images.append(img_data)
            
        return images

    def images_to_pdf(self, images, output_path):
        """
        Converts a list of images (numpy arrays or PIL Images) to a PDF.
        """
        doc = fitz.open()
        
        for img in images:
            # Convert numpy array to PIL Image if needed
            if isinstance(img, np.ndarray):
                img_pil = Image.fromarray(img)
            else:
                img_pil = img
                
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG', quality=95)
            
            page = doc.new_page(width=img_pil.width, height=img_pil.height)
            page.insert_image(page.rect, stream=img_byte_arr.getvalue())
            
        doc.save(output_path)
        print(f"Saved annotated PDF to {output_path}")
