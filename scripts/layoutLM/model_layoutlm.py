import os
import pytesseract
import json
import torch
import pandas as pd
from pdf2image import convert_from_path
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image

# Set Tesseract OCR path (modify if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Load LayoutLMv3 model
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base",apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# Convert PDF to images (if applicable)
def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

# Extract text and bounding boxes using Tesseract
def extract_text(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    width, height = image.size  # Get image dimensions

    text_data = []
    for i in range(len(data["text"])):
        if data["text"][i].strip():
            x1 = int((data["left"][i] / width) * 1000)
            y1 = int((data["top"][i] / height) * 1000)
            x2 = int(((data["left"][i] + data["width"][i]) / width) * 1000)
            y2 = int(((data["top"][i] + data["height"][i]) / height) * 1000)

            text_data.append({
                "text": data["text"][i],
                "bbox": [x1, y1, x2, y2]  # Normalized bbox
            })

    return text_data


# Run LayoutLMv3 for structured extraction
def extract_invoice_details(image, text_data):
    words = [item["text"] for item in text_data]
    boxes = [item["bbox"] for item in text_data]

    encoded_inputs = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True)
    outputs = model(**encoded_inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    # Define label mapping (adjust as needed)
    label_map = {
        0: "O",  # Outside any entity
        1: "INVOICE_NUMBER",
        2: "INVOICE_DATE",
        3: "SUPPLIER_NAME",
        4: "TOTAL_AMOUNT",
        5: "VAT_AMOUNT",
        6: "PRODUCT_ID",
        7: "PRODUCT_NAME",
        8: "QUANTITY",
        9: "COST",
        10: "VAT_RATE"
    }

    structured_data = {"products": []}
    product_entry = {}

    for word, pred in zip(words, predictions):
        label = label_map.get(pred, "O")
        if label == "O":
            continue

        if label.startswith("PRODUCT_"):
            field = label.replace("PRODUCT_", "").lower()
            product_entry[field] = word

            # If all fields are filled, save product entry
            if len(product_entry) == 5:
                structured_data["products"].append(product_entry)
                product_entry = {}
        else:
            structured_data[label.lower()] = word

    return structured_data

# Main function
def process_invoice(file_path):
    if file_path.lower().endswith(".pdf"):
        images = pdf_to_images(file_path)
    else:
        images = [Image.open(file_path)]

    all_data = []
    for image in images:
        text_data = extract_text(image)
        structured_data = extract_invoice_details(image, text_data)
        all_data.append(structured_data)

    return all_data


# Example usage
invoice_path = "..\\..\\data\\training\\Fattura-3.pdf"  # Change to your file path
extracted_data = process_invoice(invoice_path)

print(extracted_data)

# Output JSON
'''with open("invoice_data.json", "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, indent=4)

print("Invoice data saved to invoice_data.json")'''
