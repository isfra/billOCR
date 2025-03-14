import pytesseract
from pytesseract import Output
import re

from pdf2image import convert_from_path
import cv2
import numpy as np


# Convert PDF to image
pages = convert_from_path("..\\data\\pdf\\training\\Fattura-3.pdf", dpi=300)
image = np.array(pages[0])  # Convert the first page to a NumPy array

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to binarize the image
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Save the preprocessed image
cv2.imwrite("preprocessed_invoice.png", binary)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Use Tesseract with layout analysis
data = pytesseract.image_to_data(binary, output_type=Output.DICT, lang="ita", config="--psm 6")

# Extract text and bounding boxes
n_boxes = len(data['level'])
text_boxes = []
for i in range(n_boxes):
    if data['text'][i].strip():  # Ignore empty text
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        text = data['text'][i]
        text_boxes.append({"text": text, "x": x, "y": y, "w": w, "h": h})

# Sort text boxes by their y-coordinate (rows) and x-coordinate (columns)
text_boxes.sort(key=lambda b: (b["y"], b["x"]))


# Group text boxes into rows
rows = []
current_row = []
prev_y = text_boxes[0]["y"]

for box in text_boxes:
    if abs(box["y"] - prev_y) > 10:  # Adjust threshold based on your layout
        rows.append(current_row)
        current_row = []
    current_row.append(box)
    prev_y = box["y"]
rows.append(current_row)  # Add the last row

# Group text within each row into columns
table = []
for row in rows:
    row_text = " | ".join([box["text"] for box in row])  # Separate columns with "|"
    table.append(row_text)

# Print the table
for row in table:
    print(row)

# Find the header row
header_row = None
for i, row in enumerate(table):
    if "Codice" in row and "Descrizione" in row and "Quantità" in row:
        header_row = i
        break

# Extract product rows
if header_row is not None:
    products = table[header_row + 1:]  # Skip the header row
    for product in products:
        print(product)



# Parse the product rows
parsed_products = []
headers = ["Codice", "Descrizione", "Quantità", "Prezzo", "Sconto", "Importo", "Iva"]

for product in products:
    columns = re.split(r"\s*\|\s*", product)  # Split by "|"
    if len(columns) == len(headers):
        parsed_product = {headers[i]: columns[i] for i in range(len(headers))}
        parsed_products.append(parsed_product)

# Print the parsed products
for product in parsed_products:
    print(product)