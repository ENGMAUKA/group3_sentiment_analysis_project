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

## 📊 Results  
- The dataset is imbalanced (~60% neutral).  
- Logistic Regression with TF-IDF achieved strong baseline accuracy.  
- Evaluation metrics (confusion matrix + classification report) are available in the notebook.  


<img width="2400" height="1800" alt="image" src="https://github.com/user-attachments/assets/d04f3574-1489-4e92-b48f-24d34f3da61a" />


![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)

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
