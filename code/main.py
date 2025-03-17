from datasets import load_dataset
from sklearn.metrics import classification_report
import importlib
import torch
import os


# Load and/or cache the dataset
cache_file = "rottenTomatoes.data"
if os.path.exists(cache_file):
    print("Loading cached data...")
    data = torch.load(cache_file, weights_only=False)
else:
    print("Downloading data...")
    data = load_dataset("rotten_tomatoes")
    torch.save(data, cache_file)

# Have a look at the data
print("\nInput: ", data["train"]["text"][10])
print("\nOutput: ", data["train"]["label"][10])

# Iterate over the models that we are comparing
#models = ["embedding", "logisticEmbedding", "zeroShot", "chatGPT"]
models = [ "bert", "flan", "transformer" ]

for m in models:
    print("Running model: {}".format(m))
    module = importlib.import_module(m)
    y_actual = data["test"]["label"]
    y_pred   = eval('module.run(data)')
    performance = classification_report(
        y_actual, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)
