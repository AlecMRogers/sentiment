from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

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