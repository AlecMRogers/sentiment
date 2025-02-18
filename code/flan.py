from datasets import tqdm
from transformers import pipeline
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

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