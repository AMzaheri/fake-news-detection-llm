# to run: pytest tests/test_predict.py

import pytest
from app.predict_llm import predict_news

def test_predict_news_output_type():
    """Test that predict_news() returns a string."""
    text = "Breaking news: Aliens landed in Paris!"
    prediction = predict_news(text)
    assert isinstance(prediction, str), "Prediction should be a string"

def test_predict_news_output_value():
    """Test that predict_news() returns only 'REAL' or 'FAKE'."""
    text = "The stock market hits an all-time high!"
    prediction = predict_news(text)
    assert prediction in ["REAL", "FAKE"], "Prediction should be 'REAL' or 'FAKE'"

