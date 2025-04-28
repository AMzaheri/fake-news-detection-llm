import pytest
from training.llm_model_module import prepare_dataset, tokenize_dataset

def test_prepare_dataset():
    dataset = prepare_dataset()
    assert "train" in dataset and "test" in dataset, "Dataset should have train and test splits"
    assert len(dataset["train"]) > 0, "Training set is empty"
    assert len(dataset["test"]) > 0, "Test set is empty"
    labels = set(dataset["train"]["label"])
    assert labels.issubset({0, 1}), "Unexpected labels found in dataset"

def test_tokenize_dataset():
    raw_dataset = prepare_dataset()
    tokenized_dataset = tokenize_dataset(raw_dataset)
    example = tokenized_dataset["train"][0]
    assert "input_ids" in example, "Missing input_ids after tokenization"
    assert "attention_mask" in example, "Missing attention_mask after tokenization"
    assert "label" in example, "Missing label after tokenization"

