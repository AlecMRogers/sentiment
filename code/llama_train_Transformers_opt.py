from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
#from evaluate import load_metric
import numpy as np
import torch
import os
import torch.nn.utils.prune as prune

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Load Dataset
full_dataset = load_dataset("rotten_tomatoes")

tr_dataset = full_dataset["train"].train_test_split(test_size=0.95)["train"] # keep 5% of the train set.
val_dataset = full_dataset["test"].train_test_split(test_size=0.99)["test"] # keep 5% of the validation set.

# Load Tokenizer and Model
model_name = "meta-llama/Llama-3.2-1B"  # You requested 2-1B, which does not exist. 8B is the smallest.
access_token = "INSERT_YOUR_ACCESS_TOKEN_FROM_HUGGINGFACE_HERE"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, token=access_token) # rotten tomatoes has 2 labels
except OSError as e:
    print(f"Error loading model or tokenizer: {e}. Please ensure the model name is correct and you have access to it.")
    exit()

# Tokenize the Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tr_tokenized_datasets = tr_dataset.map(tokenize_function, batched=True)
val_tokenized_datasets = val_dataset.map(tokenize_function, batched=True)

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=1, #Adjust based on your GPU memory
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False, #set to true if you wish to push to hub.
    remove_unused_columns=False, #important for llama models.
    gradient_accumulation_steps=4,  # Simulate larger batch size
    fp16=True,  # Enable mixed-precision training
)

# Define Metric
#metric = load_metric("accuracy")

# def compute_metrics(eval_pred):
    #logits, labels = eval_pred
    #predictions = np.argmax(logits, axis=-1)
    #return metric.compute(predictions=predictions, references=labels)

# Rename label column and set format.
tr_tokenized_datasets = tr_tokenized_datasets.rename_column("label", "labels")
tr_tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_tokenized_datasets = val_tokenized_datasets.rename_column("label", "labels")
val_tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=tokenized_datasets["train"],
    #eval_dataset=tokenized_datasets["test"],
    train_dataset=tr_tokenized_datasets,
    eval_dataset=val_tokenized_datasets,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
)

# Train and Evaluate
print("Training.....")
trainer.train()
trainer.evaluate()
print ("Evaluation complete saving....")
# Save model
trainer.save_model("./sentiment_analysis_llama3_rotten_tomatoes")