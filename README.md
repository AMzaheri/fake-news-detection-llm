# End-to-End Fake News Detection App using LLM

Fine-tuned DistilBERT (on the ISOT dataset), deployed with Flask, Docker, and Render.

This project fine-tunes a **DistilBERT transformer model** to classify news articles as REAL or FAKE using the **ISOT Fake News Dataset**.  
It provides an interactive **Flask web app**, containerised with **Docker**, and deployed to **Render.com**.

>  *This is an educational project — model predictions reflect linguistic patterns, not factual verification.*

---

##  Project Structure

```
fake-news-detection-llm/
│
├── app/                        ← Flask app for prediction
│   ├── app.py
│   ├── predict_llm.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── training/                  ← Model training pipeline
│   ├── llm_model_module.py
│   ├── run_llm_training.py
│   └── tests/
│       ├── test_data.py
│       └── test_predict.py
│
├── data/                      ← Raw data (True.csv / Fake.csv)
├── model/                     ← Fine-tuned model (excluded from GitHub)
├── scripts/         
│   └── upload_to_hf.py        - Upload mfine tuned model to a Hugging Face repo
├── requirements.txt
├── README.md                 
└── .gitignore
```

---

##  What This Project Does

- Loads and preprocesses the ISOT Fake News Dataset
- Tokenizes and fine-tunes a DistilBERT transformer model using the 🤗 Hugging Face Trainer
- Trains two versions of the model (5-epoch and 10-epoch) for comparison
- Uploads the final model (with tokenizer) to Hugging Face Hub
- Builds an interactive web app using Flask
-  Deploys the app using Docker and Render.com
- Provides a tabbed user interface with live prediction
- Includes unit tests for dataset, tokenisation, and model functions
-  Contains a Dockerfile for local or cloud containerisation
-  Tracks training progress with TensorBoard
- Codebase is modular and documented, following best practices

---

##  Dataset Citation

This project uses the **ISOT Fake News Dataset**, created by the [ISOT Lab, University of Victoria](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/).

- **Source**: [ISOT Dataset Page](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)
---

## Model Training & Evaluation

- Tokenizer: `DistilBertTokenizerFast`
- Model: `DistilBertForSequenceClassification`
- Framework: `transformers`, `datasets`, `Trainer`
- Epochs: default = 5–10 (adjustable)
- Evaluation: accuracy, precision, recall, F1
- Live tracking: `TensorBoard` (logs/)

Training is run via:
```bash
python training/run_llm_training.py
```
Or in Kaggle notebook [see linked notebook URL](https://www.kaggle.com/code/afsanehm/fake-news-detection-with-llm-fine-tuning).

---

##  Flask Web App (app/)

Use the trained model in a Flask app to test predictions interactively.

Start it locally:
```bash
cd app
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

---

##  Unit Tests

Unit tests are in `training/tests/`:
```bash
pytest training/tests/
```
Covers:
- Data preparation
- Tokenisation
- Model structure and prediction

---

## Disclaimer

> This model was fine-tuned on historical news articles (ISOT 2017) and is not guaranteed to be accurate on modern news or factual verification. It reflects **linguistic patterns**, not factual truth.


---

## 📋 Requirements

Install everything using:
```bash
pip install -r requirements.txt
```

Key packages:
- `transformers`
- `datasets`
- `scikit-learn`
- `Flask`
- `torch`
- `pandas`

---


## Deployment with Docker

This project includes a Docker setup for running the Flask web app locally or on a server.

###  Build the Docker image
```bash
docker build -t fake-news-llm-app .
```

###  Run the app
```bash
docker run -it -p 5000:5000 fake-news-llm-app
```

Then visit: [http://localhost:5000](http://localhost:5000)

>  Make sure the `model/` folder includes the saved fine-tuned model **and tokenizer files**.
 
This Docker image is meant for local use and prototyping.

---

## Uploading Model to Hugging Face Hub

Once you have fine-tuned your model, you can upload it to your Hugging Face account using the provided script:
```bash
python scripts/upload_to_hf.py
```

### Requirements:

Make sure you are logged in:

```bash
huggingface-cli login
```
The upload script uses the directory:
```bash
model/fine_tuned_model/

```

and  pushes it to:  https://huggingface.co/afsanehm/fake-news-detection-llm


##  Future Work

- Improve model with more recent datasets
- Integrate a GPT-based zero-shot classifier
- Add feedback loop + explanation interface (e.g., LIME)
- Deploy with Hugging Face Spaces or Render

---

## ‍ Author

Afsaneh Mohammadazaheri — [GitHub](https://github.com/AMzaheri)

Please share this work. Feedback welcome!
