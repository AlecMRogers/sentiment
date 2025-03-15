from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import torch

def run(data):
    # Llama
    access_token = "INSERT_KEY_HERE"

    model_path = AutoModelForSequenceClassification.from_pretrained('meta-llama/Llama-3.2-1B', token=access_token)
    model_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B', token=access_token)
    # Set device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path.to(device)
    
    # Load model into pipeline
    pipe = pipeline(
        task='text-classification',
        model=model_path,
        tokenizer=model_tokenizer,
        top_k=1,
        device_map="auto"
    )       

    label2id = {
        0: "LABEL_0",
        1: "LABEL_1"
    }

    # Run inference
    y_pred = []
    for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
        label_id = output[0]["label"]
        label = [k for k, v in label2id.items() if v == label_id][0]
        y_pred.append(label)

    return y_pred