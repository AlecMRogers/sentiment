from datasets import load_dataset
from sklearn.metrics import classification_report
import importlib
import torch
import os

# Load our data
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


# Next we iterate over all of the models that we are comparing to categorize this dataset.
#models = ["bert", "embedding", "logisticEmbedding", "zeroShot", "flan", "chatGPT"]
models = ["embedding", "bert", "flan"]

for m in models:
    module = importlib.import_module(m)
    y_actual = data["test"]["label"]
    y_pred   = eval('module.run(data)')
    performance = classification_report(
        y_actual, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)
