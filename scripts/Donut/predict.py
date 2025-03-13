from PIL import Image
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load the fine-tuned model and processor
model_path = "./results/Donut/final_model"
model = VisionEncoderDecoderModel.from_pretrained(model_path)
processor = DonutProcessor.from_pretrained(model_path)

# Load a new invoice image
image = Image.open("..\\..\\data\\images\\training\\Fattura-2.png").convert("RGB")

# Preprocess the image
pixel_values = processor(image, return_tensors="pt").pixel_values

# Generate output
outputs = model.generate(
    pixel_values,
    max_length=512,  # Adjust based on your label length
    early_stopping=True,
)

# Decode the output
predicted_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(predicted_text)

# Parse the predicted output
def parse_predicted_output(predicted_text):
    result = {}
    # Extract fields using regex
    result["invoice_number"] = re.search(r"Invoice Number: (.+)", predicted_text).group(1)
    result["invoice_date"] = re.search(r"Invoice Date: (.+)", predicted_text).group(1)
    result["supplier_name"] = re.search(r"Supplier Name: (.+)", predicted_text).group(1)
    result["total_amount"] = float(re.search(r"Total Amount: €([\d,]+)", predicted_text).group(1).replace(",", ""))
    result["vat_amount"] = float(re.search(r"VAT Amount: €([\d,]+)", predicted_text).group(1).replace(",", ""))

    # Extract products
    products_text = re.search(r"Products:(.+)$", predicted_text, re.DOTALL).group(1)
    products = []
    for product_text in products_text.strip().split("\n"):
        product = {}
        product["id"] = re.search(r"Product ID: (.+),", product_text).group(1)
        product["name"] = re.search(r"Name: (.+),", product_text).group(1)
        product["quantity"] = int(re.search(r"Quantity: (.+),", product_text).group(1))
        product["cost"] = float(re.search(r"Cost: €([\d.]+)", product_text).group(1))
        product["vat_rate"] = int(re.search(r"VAT Rate: (.+)%", product_text).group(1))
        products.append(product)
    result["products"] = products

    return result


# Example usage
parsed_output = parse_predicted_output(predicted_text)
print(parsed_output)