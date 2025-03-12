from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load a pre-trained LayoutLM model and tokenizer
model_name = "microsoft/layoutlm-base-uncased"
model = LayoutLMForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
tokenizer = LayoutLMTokenizer.from_pretrained(model_name)

# Prepare your dataset
# Assume `dataset` is a list of dictionaries with "text" and "labels"
dataset = [
    {
        "text": "Fattura nr. 1341 del 02/11/2024\nDestinatario: FARO S.R.L.S. VIA PIAVE, 55 00187 ROMA (RM) ITALY\n...",
        "labels": {
            "invoice_number": "1341",
            "invoice_date": "02/11/2024",
            "recipient": "FARO S.R.L.S. VIA PIAVE, 55 00187 ROMA (RM) ITALY",
            "items": [...],
            "total_amount": 1294.91,
            "vat_amount": 233.51
        }
    },
    ...
]

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
    num_train_epochs=3,
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