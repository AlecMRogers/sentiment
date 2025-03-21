from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import os
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import torch
from pathlib import Path

# Custom dataset using pre-tokenized inputs
class RottenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Training function using PyTorch
def train_model(model, train_dataset, device, epochs=3, batch_size=1, learning_rate=2e-5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")
    return model


# Inference function using pre-tokenized input
def run_inference(test_encodings, model, device):
    model.eval()
    predictions = []
    test_dataset = RottenDataset(test_encodings, [0] * len(test_encodings["input_ids"]))  # Dummy labels
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions


if __name__ == "__main__":
    cache_file = "rottenTomatoes.data"
    access_token = Path('token.txt').read_text()

    # Load or cache dataset
    if os.path.exists(cache_file):
        print("Loading cached data...")
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
    else:
        print("Downloading data...")
        data = load_dataset("rotten_tomatoes")
        torch.save(data, cache_file)

    print("\nInput: ", data["train"]["text"][0])
    print("\nOutput: ", data["train"]["label"][0])

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B', token=access_token)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Padding token: {tokenizer.pad_token}")
    print(f"Padding token id: {tokenizer.pad_token_id}")
    #tokenizer.model_max_length = 512
    tokenLen = len(tokenizer)

    # Pre-tokenize training and test data
    print("Tokenizing data...")
    train_encodings = tokenizer(data["train"]["text"], truncation=True, padding=True, max_length=512)
    train_dataset = RottenDataset(train_encodings, data["train"]["label"])
    test_texts = data["test"]["text"]
    test_labels = data["test"]["label"]
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    model = AutoModelForSequenceClassification.from_pretrained('meta-llama/Llama-3.2-1B', token=access_token)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(tokenLen)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Training model...")
    model = train_model(model, train_dataset, device, epochs=1, batch_size=10, learning_rate=2e-5)

    print("Running inference on test data...")
    y_pred = run_inference(test_encodings, model, device)
    y_actual = test_labels

    performance = classification_report(
        y_actual, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

# export TOKENIZERS_PARALLELISM=true ;  python llama.py
