import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Load pre-trained TableNet model
class TableNet(torch.nn.Module):
    def __init__(self):
        super(TableNet, self).__init__()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
        self.model.fc = torch.nn.Linear(512, 2)  # Two output channels (table & column masks)

    def forward(self, x):
        return self.model(x)

# Load the model
model = TableNet()
model.load_state_dict(torch.load("tablenet.pth", map_location=torch.device("cpu")))
model.eval()


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    return transform(image).unsqueeze(0)


# Load & preprocess invoice image
image_path = "..\\data\\images\\training\\Fattura-3.png"
input_tensor = preprocess_image(image_path)

# Run prediction
with torch.no_grad():
    table_mask, column_mask = model(input_tensor)

# Convert predictions to numpy
table_mask = table_mask.squeeze().cpu().numpy()
column_mask = column_mask.squeeze().cpu().numpy()

# Resize masks to match original image size
original_image = cv2.imread(image_path)
h, w, _ = original_image.shape
table_mask = cv2.resize(table_mask, (w, h))
column_mask = cv2.resize(column_mask, (w, h))

def extract_table_regions(mask, original_image):
    mask = (mask > 0.5).astype(np.uint8)  # Convert to binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    table_crops = []  # Store detected table crops

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        table_crop = original_image[y:y+h, x:x+w]  # Extract detected table region
        table_crops.append(table_crop)  # Store the table for later use
        cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw bounding box

    return original_image, table_crops

# Get detected tables
output_image, table_crops = extract_table_regions(table_mask, original_image)

# Save image with detected tables
#cv2.imwrite("detected_table.jpg", output_image)

# Run OCR on each detected table region
for i, table_crop in enumerate(table_crops):
    table_text = pytesseract.image_to_string(table_crop, config="--psm 6")
    print(f"Extracted Text from Table {i+1}:\n", table_text)

