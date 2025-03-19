import os, torch
from datasets import tqdm
from transformers import pipeline
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report
from datasets import load_dataset

def run(data):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device="mps:0"
    )

    # Prepare our data
    prompt = "Is the following sentence positive or negative? "
    data = data.map(lambda example: {"t5": prompt + example['text']})

    # Run inference
    y_pred = []
    for output in tqdm(pipe(KeyDataset(data["test"], "t5")), total=len(data["test"])):
        text = output[0]["generated_text"]
        y_pred.append(0 if text == "negative" else 1)

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
