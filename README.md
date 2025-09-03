# Twitter Sentiment Analysis: Apple vs Google  

This project builds an **NLP model** to analyze sentiment in tweets about **Apple** and **Google** products using the [CrowdFlower dataset](https://data.world/crowdflower/brands-and-product-emotions).  
The dataset contains ~9,000 tweets labeled with human-annotated sentiment categories.  

---

## 📌 Project Overview  
- **Goal:** Classify tweets as **positive**, **negative**, or **neutral**.  
- **Methodology:** Following the **CRISP-DM framework**:  
  1. Business & Data Understanding  
  2. Data Cleaning & Preparation  
  3. Exploratory Data Analysis (EDA)  
  4. Feature Engineering (TF-IDF)  
  5. Model Building (Logistic Regression)  
  6. Evaluation & Insights  

---

## ⚙️ Key Features  
- Data cleaning (handling nulls, duplicates, irrelevant categories).  
- Text preprocessing (stopword removal, punctuation cleaning, lemmatization).  
- Label normalization (mapping multiple categories into `{positive, negative, neutral}`).  
- Feature extraction with **TF-IDF Vectorization**.  
- Model training with **Logistic Regression**.  
- Model evaluation using **accuracy, precision, recall, F1-score, confusion matrix**.  

---

## 📂 Key Repository Areas  
```
├── data/
│   └── judge-1377884607_tweet_product_company.csv   
├── notebooks/
│   └── sentiments_consolidated.ipynb (Main analysis notebook)
├── images
│   └── functionality plot                          
├── README.md
└── Presentation PDF
```

---

## 🚀 Getting Started  

### 1. Business Problem:

In today’s highly competitive technology industry, customer perception and sentiment play a crucial role in shaping brand reputation and influencing purchasing decisions. Apple and Google are two of the most recognized technology companies worldwide, and their products often generate strong opinions on social media platforms such as Twitter.

The goal of this project is to build a Natural Language Processing (NLP) model that can automatically classify the sentiment of Tweets related to Apple and Google products. By analyzing over 9,000 Tweets labeled as positive, negative, or neutral, the model will provide insights into how consumers feel about these brands and their products.


### 2. Project Goals:

To build a model that can rate the sentiments of a Tweet based on its content.


### 3. Dataset:

The dataset employed in the study was downloaded from https://data.world/crowdflower/brands-and-product-emotions/file/judge-1377884607_tweet_product_company.csv

### 4. Methodology:

The adopted structure for the project was CRISP-DM that entails undertaking Business understanding; Data Understanding; Data Preparation; Data Cleaning and Exploratory Data Analysis (EDA); Modelling; Conclusion and Recommendations.  

---
## Modeling:
### 1).Dummy Classifier Model

                precision    recall  f1-score   support

    Negative       0.08      0.08      0.08       142
    Positive       0.34      0.33      0.33       743
     Neutral       0.60      0.61      0.60      1344

    accuracy                           0.48      2229
   macro avg       0.34      0.34      0.34      2229
weighted avg       0.48      0.48      0.48      2229

<img width="560" height="495" alt="image" src="https://github.com/user-attachments/assets/3305a71c-809d-4bec-8005-c00f2c99511c" />

Accuracy: 0.4809
Weighted Recall: 0.4809
Weighted F1 Score: 0.4808

Confusion Matrix Insights - Dummy Classifier:

Negative tweets: 12 correctly classified, 97 misclassified as neutral, 43 as positive. Neutral tweets: 815 correctly classified out of 1344, with some misclassified as positive or negative. Positive tweets: 245 correctly classified, 432 misclassified as neutral, 44 as negative.

The baseline model performed poorly with an accuracy, weighted recall score and weighted f1 score of 48.09%. The model struggled to predict the negative emotions with a precision score of 8%. This could likely be due class imbalances originating from the available dataset. The data is very imbalanced which explains the base model performance of close to 50%

### 2).Logistic Regression
Logistic Regression Test data Model Score:
              precision    recall  f1-score   support

    Negative       0.60      0.23      0.33       142
    Positive       0.61      0.52      0.56       743
     Neutral       0.72      0.83      0.77      1344

    accuracy                           0.69      2229
   macro avg       0.65      0.52      0.55      2229
weighted avg       0.68      0.69      0.67      2229

<img width="569" height="495" alt="image" src="https://github.com/user-attachments/assets/ef7bebee-7652-4fcb-925d-41035e5793c7" />

Accuracy: 0.6878
Weighted Recall: 0.6878
Weighted F1 Score: 0.6735

Confusion Matrix Insights: - Logistic Regression

Negative tweets: 37 correctly classified, 9 misclassified as neutral, 12 as positive. Recall improvement is notable. Neutral tweets: 1116 correctly classified out of 1344, with some misclassified as positive or negative. Positive tweets: 385 correctly classified, 219 misclassified as neutral, 26 as negative.

The logistic regression model improved over the dummy model with an accuracy score of 68.78% compared to 48.09%. The Neutral class had the highest precision, recall and f1_score of 72%, 83% and 77% respectively. The Neutral class had the lowest recall (23%) and f1-score (33%).

### 3).Decision Tree Model

<img width="569" height="495" alt="image" src="https://github.com/user-attachments/assets/63004956-47b0-4ac0-8a57-76517c77a56c" />

Accuracy: 0.6339
Weighted Recall: 0.6339
Weighted F1 Score: 0.6058
Best estimator score: 0.6339

Confusion Matrix Insights - Decision Tree:

Negative tweets: 12 correctly classified, 12 misclassified as neutral, 4 as positive. Recall improvement is negligible. Neutral tweets: 1109 correctly classified out of 1344, with some misclassified as positive or negative. Positive tweets: 292 correctly classified, 223 misclassified as neutral, 25 as negative.

The decision tree model accuracy score improved to 63% compared to the dummy model accuracy score of 48.09% but performed poorly compared to the logistic regression accuracy of 68.78%. Like the logistic regression model, the model struggled to predict the negative class with a precision score of 43% and recall of 8%. This could likely be due class imbalances originating from the available dataset. The neutral class had the highest precision at 67%.

### 4).Random Forest
Random Forest Test data Model Score:
              precision    recall  f1-score   support

    Negative       0.63      0.18      0.28       142
    Positive       0.64      0.43      0.52       743
     Neutral       0.69      0.87      0.77      1344

    accuracy                           0.68      2229
   macro avg       0.66      0.50      0.52      2229
weighted avg       0.67      0.68      0.66      2229

<img width="569" height="495" alt="image" src="https://github.com/user-attachments/assets/815f6f39-2e98-41e3-b942-20fb0049a7be" />

Accuracy: 0.6801
Weighted Recall: 0.6801
Weighted F1 Score: 0.6555
Best estimator score: 0.6801
Optimized Random Forest Test Set Score: 0.680

Confusion Matrix Insights - Random Forest:

Negative tweets: 26 correctly classified, 10 misclassified as neutral, 5 as positive. Neutral tweets: 1167 correctly classified out of 1344, with some misclassified as positive or negative. Positive tweets: 323 correctly classified, 167 misclassified as neutral, 14 as negative.

The random forest model accuracy score improved to 68.01% compared to the dummy model accuracy score of 48.09% and decision tree model accuracy score improved to 63% but performed poorly compared to the logistic regression accuracy of 68.78%.

## 📊 Results  
- The dataset is imbalanced (~60% neutral).  
- Logistic Regression with TF-IDF achieved strong baseline accuracy.  
- Evaluation metrics (confusion matrix + classification report) are available in the notebook.  


<img width="2400" height="1800" alt="image" src="https://github.com/user-attachments/assets/d04f3574-1489-4e92-b48f-24d34f3da61a" />


<img width="5400" height="8400" alt="image" src="https://github.com/user-attachments/assets/d7b6d99a-949c-4171-8efd-ddade77b27b4" />


---

## Conclusion:

Logistic Regression and SVC improved recall for the negative class, making the model less biased toward neutral. Positive sentiment performance is moderate, while neutral remains strong.

## Recommendations:

- Further threshold optimization per class.
- Feature engineering (e.g., combining char- and word-level n-grams).
- Addressing class imbalance to improve negative and positive precision.

## 🔮 Future Work  
- Try **deep learning models** (LSTM, BERT).  
- Apply **hyperparameter tuning** for better performance.  
- Track **sentiment trends over time**.  

---

## 🛠️ Key Requirements  
- Python 3.8+  
- pandas  
- scikit-learn  
- nltk  
- matplotlib  
- seaborn  
- jupyter  

---

## 📜 License  
This project is open-source under the MIT License.  
