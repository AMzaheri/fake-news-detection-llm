# llm_model_module.py

from transformers import pipeline

import os
import pandas as pd
from datasets import Dataset

#---------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "model")  # <== Fine-tuned model 
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")  
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")      

#---------------------------------------------------
# Load the pre-trained model from Hugging Face
# This will download on first run and cache locally
classifier = pipeline("text-classification", \
             model="distilbert-base-uncased-finetuned-sst-2-english")

#---------------------------------------------------
from datasets import Dataset

def prepare_dataset():
    # Load both parts of ISOT dataset
    true_df = pd.read_csv(os.path.join(PROJECT_ROOT, \
                          "data", "True.csv"))
    fake_df = pd.read_csv(os.path.join(PROJECT_ROOT, \
                          "data", "Fake.csv"))

    # Add labels
    true_df["label"] = 1  # REAL
    fake_df["label"] = 0  # FAKE

    # Keep only the 'text' and 'label' columns
    true_df = true_df[["text", "label"]]
    fake_df = fake_df[["text", "label"]]

    # Combine and shuffle
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert to Hugging Face Dataset and split
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)

    return dataset  # A DatasetDict with 'train' and 'test'

#---------------------------------------------------
from transformers import DistilBertTokenizerFast

def tokenize_dataset(dataset):
    '''
    Load the tokenizer from Hugging Face.
    Define a preprocessing function that:
    Tokenize the text field.
    Applly truncation and padding up to 512 tokens.
    Appliy this to the full dataset using map.
    Set the format of the returned dataset for PyTorch
    '''
    tokenizer = DistilBertTokenizerFast.from_pretrained(\
                 'distilbert-base-uncased')
    def tokenize_fn(example):
        return tokenizer(
            example['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=512
        )
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    # This part is essential for model training with PyTorch
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'label']
    )
    return tokenized_dataset
#---------------------------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def train_model(tokenized_dataset):
    from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Manually split into train and test sets (80% train, 20% test)
    #from sklearn.model_selection import train_test_split
    #train_dataset, eval_dataset = train_test_split(tokenized_dataset, test_size=0.2)
  
    tokenized_dataset = tokenized_dataset["train"]  # Make sure you have a "train" split
    train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.2).values()
 
    # Split into training and test sets
    #train_test = tokenized_dataset.train_test_split(test_size=0.2)
    #train_dataset = train_test["train"]
    #eval_dataset = train_test["test"]

    model = DistilBertForSequenceClassification.from_pretrained(\
            "distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        logging_dir=LOGS_DIR,
        num_train_epochs=1,  # a quick test run
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy ="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # Save fine-tuned model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_DIR)

    return model
