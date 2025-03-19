import os, torch
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report
from datasets import load_dataset

def run(data):
    # Path to our HF model
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Load model into pipeline
    pipe = pipeline(
        model=model_path,
        tokenizer=model_path,
        return_all_scores=True,
        device="mps:0"  # on mac, or "cuda:0" (for ness), also "cpu" and several other alternatives
    )

    # Run inference
    y_pred = []
    for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
        negative_score = output[0]["score"]
        positive_score = output[2]["score"]
        assignment = np.argmax([negative_score, positive_score])
        y_pred.append(assignment)

    return y_pred

# Standalone execution entry point
if __name__ == "__main__":
    cache_file = "rottenTomatoes.data"

    # Load or cache the pre-trained Word2Vec model
    if os.path.exists(cache_file):
        print("Loading cached data...")
        data = torch.load(cache_file, weights_only=False)
    else:
        print("Downloading data...")
        data = load_dataset("rotten_tomatoes")
        torch.save(data, cache_file)

    # Have a look at the data
    print("\nInput: ", data["train"]["text"][0])
    print("\nOutput: ", data["train"]["label"][0])

    print("Running model...")
    y_actual = data["test"]["label"]
    y_pred = run(data)
    performance = classification_report(
        y_actual, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)
