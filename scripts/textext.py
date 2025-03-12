from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def extract_text_from_pdf(pdf_path):
    # Convert PDF pages to images
    images = convert_from_path(pdf_path,poppler_path=r'C:\\Program Files\\poppler-24.08.0\\Library\\bin')
    text = ""
    for image in images:
        # Extract text from each image
        text += pytesseract.image_to_string(image)
    return text


# Example usage
pdf_path = "..\\data\\training\\Fattura-1.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
print(extracted_text)