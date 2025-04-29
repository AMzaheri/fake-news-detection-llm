# run_predict.py

from predict_llm import predict_news

if __name__ == "__main__":
    example_text = "Scientists discover a new species of bird in the Amazon rainforest."
    prediction = predict_news(example_text)
    print(f"Prediction: {prediction}")

