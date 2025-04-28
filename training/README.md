# Training Module for Fake News Detection (LLM Version)

This folder contains scripts to prepare the dataset, fine-tune a pre-trained DistilBERT model, and save the fine-tuned model for downstream prediction.

---

## Structure

- `llm_model_module.py`: Core module for dataset preparation, tokenization, model training, and evaluation.
- `run_llm_training.py`: Entry-point script to run full training easily.
- `logs/`: TensorBoard logs will be stored here.
- `results/`: Intermediate results (checkpoints) during training.
- `../model/`: Final fine-tuned model is saved outside this folder for easy access.


## How to Train Locally

cd training
python run_llm_training.py

This will:
 
 -Load and preprocess the ISOT dataset.
 -Fine-tune a DistilBERT model for fake news classification.
 -Save the fine-tuned model to ../model/.

## Important Notes


 -Training uses CPU or GPU depending on your device availability.
 -Logs can be viewed for training progress (logs/).
 -The model expects news text input and outputs "REAL" or "FAKE" labels.

## Requirements

 -transformers
 -datasets
 -torch
 -scikit-learn
 -pandas
(See the project-level requirements.txt for the full list.)

## Unit Tests

Unit tests for dataset loading and core training functions are located inside the tests/ folder.
To run all tests:

pytest tests/
