#--------------
# run the flask app locally 
# python app_local_flask.py
#Visit http://127.0.0.1:5000 in your browser. This should work if the model is in model/fine_tuned_model.
#----------------------------

from flask import Flask, render_template, request
from predict_llm_local_flask import predict_news
import os

os.environ["FLASK_RUN_HOST"] = "0.0.0.0"
os.environ["FLASK_RUN_PORT"] = "5000"


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        news_text = request.form['news_text']
        prediction = predict_news(news_text)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

