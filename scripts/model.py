from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import dataset

# Load a pre-trained LayoutLM model and tokenizer
model_name = "microsoft/layoutlm-base-uncased"
model = LayoutLMForTokenClassification.from_pretrained(model_name, num_labels=6)
tokenizer = LayoutLMTokenizer.from_pretrained(model_name)

# Prepare your dataset
# Assume `dataset` is a list of dictionaries with "text" and "labels"
dataset = dataset.dataset_dict

# Convert dataset to Hugging Face Dataset format
hf_dataset = Dataset.from_dict({
    "text": [item["text"] for item in dataset],
    "labels": [item["labels"] for item in dataset],
})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
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