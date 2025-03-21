
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import os
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import torch


# Custom dataset for Rotten Tomatoes data
class RottenDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        # Convert all encodings to torch tensors and add the label
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Training function using PyTorch
def train_model(model, tokenizer, train_data, device, epochs=3, batch_size=1, learning_rate=2e-5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_texts = train_data["text"]
    train_labels = train_data["label"]

    train_dataset = RottenDataset(train_texts, train_labels, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            # Move the batch to the device
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


# Inference function using Hugging Face's pipeline
def run_inference(data, model, tokenizer):
    # Create a pipeline for text classification using the fine-tuned model
    pipe = pipeline(
        task='text-classification',
        model=model,
        tokenizer=tokenizer,
        top_k=1,
        device_map="auto"
    )
    # Define mapping from label string to numeric label (adjust if necessary)
    label2id = {
        0: "LABEL_0",
        1: "LABEL_1"
    }
    y_pred = []
    for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
        label_id = output[0]["label"]
        # Find the numeric label corresponding to the label string
        label = [k for k, v in label2id.items() if v == label_id][0]
        y_pred.append(label)
    return y_pred


# Standalone execution entry point
if __name__ == "__main__":
    cache_file = "rottenTomatoes.data"

    # Load or cache the dataset
    if os.path.exists(cache_file):
        print("Loading cached data...")
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
    else:
        print("Downloading data...")
        data = load_dataset("rotten_tomatoes")
        torch.save(data, cache_file)

    # Display sample data from the training set
    print("\nInput: ", data["train"]["text"][0])
    print("\nOutput: ", data["train"]["label"][0])

    # Access token for the Hugging Face model
    access_token = ""

    # Load the model and tokenizer (for sequence classification)
    model = AutoModelForSequenceClassification.from_pretrained('meta-llama/Llama-3.2-1B', token=access_token)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B', token=access_token)

    #### THIS TOKENIZER-SETTING CODE IS PROBABLY NOT CORRECT.
    # I'm not sure how to set the token, which is necessary to separate the batches in an LLM context window,
    # so until it is fixed we have to use a batch size of 1.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # Set device (GPU if available, else CPU) and move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model on the training dataset
    print("Training model...")
    model = train_model(model, tokenizer, data["train"], device, epochs=1, batch_size=1, learning_rate=2e-5)

    # Run inference on the test dataset using the fine-tuned model
    print("Running inference on test data...")
    y_actual = data["test"]["label"]
    y_pred = run_inference(data, model, tokenizer)

    # Evaluate performance
    performance = classification_report(
        y_actual, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)