#  Fake News Detection Using LLM — Local Flask App

This project fine-tunes a **DistilBERT transformer model** on the [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/), builds an end-to-end pipeline, and serves predictions using a **Flask web app** running locally.

---

##  Project Structure

```
fake-news-detection-llm/
│
├── app/                        ← Flask app for local predictions
│   ├── app_local_flask.py     ← Run this to launch local app
│   ├── predict_llm_local_flask.py
│   ├── templates/index.html
│   └── static/style.css
│
├── training/                  ← Model training pipeline
│   ├── llm_model_module.py
│   ├── run_llm_training.py
└── tests/
│   ├── test_data.py
│   └── test_predict.py
│
├── scripts/
│   └── upload_to_hf.py        ← Upload fine-tuned model to Hugging Face
│
├── data/                      ← ISOT dataset (True.csv, Fake.csv)
├── model/                     ← Fine-tuned model & tokenizer (excluded from GitHub)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## What This Project Does

- Loads and preprocesses the ISOT dataset (real/fake news)
- Fine-tunes a DistilBERT transformer model with Hugging Face Trainer
- Saves both model weights and tokenizer to `model/fine_tuned_model/`
- Offers two versions of the model: 5-epoch and 10-epoch
- Builds a local Flask web app with:
  - Prediction tab
  - Disclaimer tab
  - Model comparison (5 vs 10 epochs)
- Uploads the model to Hugging Face Hub

---

## Training the Model

You can run the fine-tuning and save the model locally:

```bash
python training/run_llm_training.py
```

The trained model is saved to:

```bash
model/fine_tuned_model/
```

You can also use the Kaggle notebook:

[Kaggle Notebook](https://www.kaggle.com/code/afsanehm/fake-news-detection-with-llm-fine-tuning)

Then download the model to:

```bash
model/fine_tuned_model/
```

Alternatively, download it directly from [Hugging Face Hub](https://huggingface.co/afsanehm/fake-news-detection-llm)

You can also upload your own version:

```bash
huggingface-cli login
python scripts/upload_to_hf.py
```

---

## Unit Tests

Run tests from the root:

```bash
pytest training/tests/
```

Tests cover:
- Dataset loading and labelling
- Tokenisation
- Model prediction shapes

---

## Run the Flask App Locally

Launch the app:

```bash
python app/app_local_flask.py
```

Then open in browser:

```
http://127.0.0.1:5000
```

### The App Interface Has:
- **Tab 1**: Predict — enter a news article to classify as REAL or FAKE
- **Tab 2**: Disclaimer — limitations of the model
- **Tab 3**: Model Comparison — 5 vs 10 epoch notes

---

## Dataset

This project uses the [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/).

---

##  Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Key packages:
- `transformers`, `datasets`, `torch`
- `scikit-learn`, `pandas`
- `Flask`

---

## Disclaimer

> This model was trained on historical (2017) news articles. It may reflect **stylistic patterns**, not truth.  
> It is not intended for factual verification or real-world misinformation detection.

---

