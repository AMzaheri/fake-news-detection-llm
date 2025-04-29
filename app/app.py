
from flask import Flask, render_template, request
from predict_llm import predict_news
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        news_text = request.form['news_text']
        prediction = predict_news(news_text)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

