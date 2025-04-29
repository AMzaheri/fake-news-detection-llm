# run_predict.py

from predict_llm import predict_news

if __name__ == "__main__":
    example_text = "Breaking news: Aliens landed in Paris!"
    prediction = predict_news(example_text)
    print(f"Prediction: {prediction}")

