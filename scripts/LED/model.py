from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
import dataset_LED
import pandas as pd
from transformers import LEDTokenizer, LEDForConditionalGeneration

# Prepare your dataset
# Assume `dataset` is a list of dictionaries with "text" and "labels"
dataset = dataset_LED.dataset_dict

# Convert the dataset to Hugging Face Dataset format
hf_dataset = Dataset.from_dict({
    "text": [invoice["text"] for invoice in dataset],
    "labels": [invoice["labels"] for invoice in dataset],
})


def preprocess_function(examples):
    # Input: Raw invoice text
    inputs = [f"parse invoice: {text}" for text in examples["text"]]

    # Target: Flattened labels as a string
    targets = []
    for label in examples["labels"]:
        # Flatten the labels into a structured string
        products_str = ", ".join(
            [
                f"id: {p['id']}, name: {p['name']}, quantity: {p['quantity']}, cost: {p['cost']}, vat_rate: {p['vat_rate']}"
                for p in label["products"]]
        )
        flat_label = (
            f"id: {label['id']}, "
            f"date: {label['date']}, "
            f"supplier: {label['supplier']}, "
            f"amount: {label['amount']}, "
            f"vat_amount: {label['vat_amount']}, "
            f"products: [{products_str}]"
        )
        targets.append(flat_label)

    # Tokenize inputs and targets
    model_inputs = tokenizer(
        inputs,
        max_length=16384,  # LED supports up to 16,384 tokens
        truncation=True,  # Truncate if necessary
        padding="max_length",
        return_tensors="pt",  # Return PyTorch tensors
    )

    # Tokenize targets (if needed)
    labels = tokenizer(
        targets,
        max_length=1024,  # Adjust based on your target sequence length
        truncation=True,  # Truncate if necessary
        padding="max_length",
        return_tensors="pt",  # Return PyTorch tensors
    )

    # Add labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Load LED tokenizer and model
model_name = "allenai/led-base-16384"
tokenizer = LEDTokenizer.from_pretrained(model_name)
model = LEDForConditionalGeneration.from_pretrained(model_name)

# Tokenize the dataset
tokenized_dataset = hf_dataset.map(preprocess_function, batched=True)

# Convert lists to PyTorch tensors
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="../results/LED",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduce batch size for long sequences
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,

)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./results/LED/final_model")
tokenizer.save_pretrained("./results/LED/final_model")




