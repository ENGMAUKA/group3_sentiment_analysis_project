# Apple & Google Twitter Sentiment Analysis

This project builds an NLP model to analyze sentiment in tweets about Apple and Google products using the CrowdFlower dataset.

## Overview

The workflow:
1. Load and inspect the dataset (`judge_1377884607_tweet_product_company.csv`).
2. Normalize sentiment labels into **positive**, **negative**, or **neutral**.
3. Filter tweets related to Apple or Google products/brands.
4. Split into training and validation sets.
5. Train a pipeline: **TF-IDF (1–2 grams) + Logistic Regression**.
6. Evaluate the model and plot results (accuracy, F1, confusion matrix).
7. Save the trained model and predictions.

## Results

- **Validation Accuracy:** ~0.83  
- **Macro F1:** ~0.44 (due to class imbalance: many more positive tweets than neutral/negative).

Confusion matrix and label distribution plots are generated inside the notebook.

## Files

- `apple_google_sentiment_analysis.ipynb` — Jupyter Notebook with the full workflow.  
- `apple_google_twitter_sentiment.joblib` — Trained model pipeline (TF-IDF + Logistic Regression).  
- `apple_google_predictions.csv` — Predictions for the Apple/Google subset of the dataset.  
- `judge_1377884607_tweet_product_company.csv` — Original dataset (not included here, but required).  
- `README.md` — This file.

## Usage

### Running the Notebook
Open `apple_google_sentiment_analysis.ipynb` in **VS Code** (with the Python and Jupyter extensions) or in Jupyter Lab/Notebook. Run all cells step by step.

### Using the Model in Python

```python
import joblib

# Load trained pipeline
pipe = joblib.load("apple_google_twitter_sentiment.joblib")

# Example predictions
texts = ["Love my new iPhone!", "Pixel battery is terrible."]
print(pipe.predict(texts))
```

Expected output (approximate):
```
['positive' 'negative']
```

## Next Steps

- Handle class imbalance (oversampling, undersampling, or advanced loss functions).  
- Experiment with alternative models (e.g., Linear SVM, Random Forest, or Transformers like BERT).  
- Improve neutral class detection (currently underrepresented).  

---
