from transformers import DonutProcessor, VisionEncoderDecoderModel
import dataset_donut
from datasets import Dataset

from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AdamW
import torch
import json
from torch.utils.data import Dataset, DataLoader


# Load Donut processor and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Set the decoder_start_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")

# Set the pad_token_id
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id  # Use EOS token as a fallback
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Custom Dataset Class
class DonutDataset(Dataset):
    def __init__(self, dataset_list, processor):
        self.dataset = dataset_list  # This is a list of dictionaries
        self.processor = processor

    def __len__(self):
        return len(self.dataset)  # Return the length of the list

    def __getitem__(self, idx):
        # Get the item at the specified index
        item = self.dataset[idx]

        # Load image
        image = Image.open(item["image"]).convert("RGB")

        # Preprocess image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)  # Remove batch dimension

        # Prepare labels
        labels = json.dumps(item["labels"])
        labels = self.processor.tokenizer(
            labels,
            max_length=1024,  # Adjust based on your label length
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)  # Remove batch dimension

        return {"pixel_values": pixel_values, "labels": labels}


# Create the custom dataset
custom_dataset = DonutDataset(dataset_donut.dataset_dict, processor)

# Create the DataLoader
batch_size = 2  # Adjust based on your memory constraints
train_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)


# Lightning Module
class DonutModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return train_dataloader


# Initialize the model
donut_model = DonutModel(model)

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="../results/Donut",  # Directory to save checkpoints
    filename="donut-invoice-parser-{epoch}-{train_loss:.2f}",  # Checkpoint file name
    monitor="train_loss",  # Metric to monitor
    mode="min",  # Save the model when the monitored metric is minimized
    save_top_k=1,  # Save only the best model
)

# Define the Trainer
trainer = pl.Trainer(
    max_epochs=2,
    accelerator="cpu",  # Use CPU for training
    callbacks=[checkpoint_callback],  # Add the checkpoint callback
)

# Fine-tune the model
trainer.fit(donut_model)

# Save the model and processor after fine-tuning
model.save_pretrained("../results/Donut")
processor.save_pretrained("../results/Donut")