from transformers import DonutProcessor, VisionEncoderDecoderModel
import dataset_donut
from datasets import Dataset

from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AdamW


# Load Donut processor and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

dataset = dataset_donut.dataset_dict

# Convert to Hugging Face Dataset
hf_dataset = Dataset.from_dict({
    "image": [item["image"] for item in dataset],
    "label": [item["label"] for item in dataset],
})


def preprocess_function(examples):
    # Load images
    images = [Image.open(image_path).convert("RGB") for image_path in examples["image"]]

    # Prepare labels
    labels = [str(label) for label in examples["label"]]

    # Preprocess images and tokenize labels
    pixel_values = processor(images, return_tensors="pt").pixel_values
    labels = processor.tokenizer(
        labels,
        max_length=512,  # Adjust based on your label length
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids

    return {"pixel_values": pixel_values, "labels": labels}


# Apply preprocessing
tokenized_dataset = hf_dataset.map(preprocess_function, batched=True)


class DonutModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)

# Initialize the model
donut_model = DonutModel(model)

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="./results/Donut",  # Directory to save checkpoints
    filename="donut-invoice-parser-{epoch}-{train_loss:.2f}",  # Checkpoint file name
    monitor="train_loss",  # Metric to monitor
    mode="min",  # Save the model when the monitored metric is minimized
    save_top_k=1,  # Save only the best model
)

# Define the Trainer
trainer = pl.Trainer(
    max_epochs=3,
    gpus=1,  # Use GPU if available
    callbacks=[checkpoint_callback],  # Add the checkpoint callback
)

# Fine-tune the model
trainer.fit(donut_model, tokenized_dataset)


