import os
from pdf2image import convert_from_path


def pdf_to_images(folder_path, output_folder):
    """Converts the first page of all PDFs in a folder to images."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            images = convert_from_path(pdf_path, first_page=1, last_page=1)  # Extract only the first page

            pdf_name = os.path.splitext(filename)[0]
            image_path = os.path.join(output_folder, f"{pdf_name}.png")
            images[0].save(image_path, "PNG")
            print(f"Saved: {image_path}")


if __name__ == "__main__":
    input_folder = "..\\..\\data\\pdf\\test"  # Change this to your PDF folder path
    output_folder = "..\\..\\data\\images\\test"  # Change this to where images should be saved
    pdf_to_images(input_folder, output_folder)
