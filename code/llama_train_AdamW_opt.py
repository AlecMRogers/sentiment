import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.optim import AdamW
import os

access_token = "INSERT_YOUR_ACCESS_TOKEN_FROM_HUGGINGFACE_HERE"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load Dataset and Tokenizer
dataset = load_dataset("rotten_tomatoes")
model_name="meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create DataLoaders
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=1, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=1)

# Load Model and Optimizer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training Loop
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.train()

epochs = 3
for epoch in range(epochs):
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device).squeeze(1) 
        attention_mask = batch["attention_mask"].to(device).squeeze(1) 
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    # Evaluation Loop
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device).squeeze(1)
            attention_mask = batch["attention_mask"].to(device).squeeze(1)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_eval_loss += outputs.loss.item()

    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    print(f"Epoch {epoch+1}, Avg Eval Loss: {avg_eval_loss}")
    model.train() 

#Save the model.
model.save_pretrained("./custom_trained_model")