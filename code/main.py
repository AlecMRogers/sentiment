from datasets import load_dataset
from sklearn.metrics import classification_report
import importlib

# Load our data
data = load_dataset("rotten_tomatoes")

# Have a look at the data
data["train"][0, -1]

# Next we iterate over all of the models that we are comparing to categorize this dataset.
models = ["bert", "embedding", "logisticEmbedding", "zeroShot", "flan", "chatGPT"]


for m in models:
    module = importlib.import_module(m)
    y_actual = data["test"]["label"]
    y_pred   = eval('module.run(data)')
    performance = classification_report(
        y_actual, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)
