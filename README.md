---
title: Fake News Detection LLM
emoji: 📰
colorFrom: red
colorTo: pink
sdk: streamlit
sdk_version: "1.32.0"
app_file: app_streamlit/app.py
pinned: false
license: mit
tags:
  - streamlit
  - fake-news
  - llm
  - transformers
  - huggingface
---

#  Fake News Detection Using LLM


This project fine-tunes a **DistilBERT transformer model** on the [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/), builds an end-to-end pipeline, and serves predictions through both a **Flask app** (v1.0) and a **Streamlit web app** (v2.0).

> This repo includes **two versions**:
> - v1.0: Flask web app for local use
> - v2.0: Streamlit interface (deployable to Hugging Face Spaces)

---

## Project Structure

```
fake-news-detection-llm/
│
├── app_flask/                 ← Flask app (v1.0)
│   ├── app_local_flask.py
│   └── predict_llm_local_flask.py
│
├── app_streamlit/            ← Streamlit app (v2.0)
│   ├── app.py
│   ├── utils.py
│   ├── tabs/
│   │   ├── predict_tab.py
│   │   ├── disclaimer_tab.py
│   │   └── dashboard_tab.py
│   └── run_app_streamlit.sh  ← run app locally
│
├── training/                 ← Model training pipeline
│   ├── llm_model_module.py
│   └── run_llm_training.py
│
├── tests/                    ← Unit tests
│   ├── test_data.py
│   └── test_predict.py
│
├── scripts/
│   └── upload_to_hf.py
│
├── data/                     ← ISOT dataset (True.csv, Fake.csv)
├── model/                    ← Fine-tuned model (excluded from GitHub)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## What This Project Does

- Loads and preprocesses the ISOT dataset (real/fake news)
- Fine-tunes a DistilBERT transformer model with Hugging Face Trainer
- Saves both model weights and tokenizer to `model/fine_tuned_model/`
- Builds two interfaces for prediction:
  - Flask-based (v1.0)
  - Streamlit-based (v2.0)
- Uploads model to Hugging Face Hub
- Includes unit tests, modular structure, and documentation

---

## Training the Model

Run training locally:

```bash
python training/run_llm_training.py
```

The trained model is saved to:

```bash
model/fine_tuned_model/
```

Alternatively, use the Kaggle notebook:

 [Kaggle Notebook](https://www.kaggle.com/code/afsanehm/fake-news-detection-with-llm-fine-tuning)

Or download the model from Hugging Face:

 [Model on Hugging Face](https://huggingface.co/afsanehm/fake-news-detection-llm)

You can also upload your own model:

```bash
huggingface-cli login
python scripts/upload_to_hf.py
```

---

## Unit Tests

Run tests from the root:

```bash
pytest tests/
```

Covers:
- Dataset preparation
- Tokenisation
- Model prediction

---

## Local App Options

### Version 1: Run Flask App Locally

```bash
python app_flask/app_local_flask.py
```
Then visit:
```
http://127.0.0.1:5000
```

### Version 2: Run Streamlit App Locally

```bash
PYTHONPATH=. streamlit run app_streamlit/app.py
```
OR using the helper script:
```bash
bash app_streamlit/run_app_streamlit.sh
```

Tabs include:
- Predict
- Disclaimer
- Dashboard (coming soon)

---

This app is also deployed on Hugging Face Spaces: [View Live Demo](https://huggingface.co/spaces/afsanehm/fake-news-detection-llm)


## Dataset Citation

This project uses the [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/).


---

## Requirements

Install all dependencies:
```bash
pip install -r requirements.txt
```

Key packages:
- `transformers`, `datasets`, `torch`
- `scikit-learn`, `pandas`
- `Flask`, `streamlit`

---

## Disclaimer

> This model was trained on historical news (ISOT). It captures **linguistic patterns** and **structure**, not factual truth. It is not a fact-checking tool.

---

## ‍♀️ Author

**Afsaneh Mohammadazaheri**  
[GitHub](https://github.com/AMzaheri)  
[Kaggle Notebook](https://www.kaggle.com/code/afsanehm/fake-news-detection-with-llm-fine-tuning)

---

## Project Phases

- **v1.0**: Flask app with local model inference
- **v2.0**: Streamlit interface, Hugging Face-hosted model, expandable for dashboards or deployment


