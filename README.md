fake-news-bot-detection
Developed a machine learning-based system using NLP to classify fake news and detect bot accounts through behavioral analysis, improving misinformation detection.
fake-news-bot-detection/
│
├── data/
│   └── news_data.csv
│
├── fake_news_detection.py
├── bot_detection.py
│
├── requirements.txt
└── README.me

File: data/news_data.csv
text,label
"Government announces new education policy",0
"India wins cricket match today",0
"Scientists discover new planet",0
"Drinking salt water cures cancer",1
"Aliens landed in Delhi yesterday",1
"Miracle herb cures all diseases",1
0 = Real News
1 = Fake News


Fake News Detection Code
📄 File: fake_news_detection.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("data/news_data.csv")

X = data["text"]
y = data["label"]

# Convert text into numerical format
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Test input
test_news = ["Drinking hot water cures cancer"]
test_vector = vectorizer.transform(test_news)

prediction = model.predict(test_vector)

if prediction[0] == 1:
    print("Fake News")
else:
    print("Real News")
output:Fake News

# Simple Bot Detection using behavior rules

posts_per_day = 250
active_hours = 24

if posts_per_day > 100 and active_hours > 20:
    print("Bot Account")
else:
    print("Human Account")

    output: Bot Account

Requirements File
numpy
pandas
scikit-learn
# Fake News & Bot Detection Platform

This project is a Machine Learning-based system that detects fake news content
and identifies automated bot accounts using simple NLP and behavior analysis.

## Features
- Fake news classification (Real / Fake)
- Bot detection using activity patterns
- Beginner-friendly implementation
- Clean and simple code structure

## Technologies Used
- Python
- Machine Learning
- Natural Language Processing (NLP)
- Scikit-learn
- Pandas

## Project Structure
fake-news-bot-detection/
│
├── data/
│   └── news_data.csv
│
├── fake_news_detection.py
├── bot_detection.py
├── requirements.txt
└── README.md

## How to Run the Project

1. Install dependencies  
   pip install -r requirements.txt

2. Run fake news detection  
   python fake_news_detection.py

3. Run bot detection  
   python bot_detection.py

## Sample Output
Fake News  
Bot Account

## Use Cases
- Social media moderation
- Misinformation control
- Digital trust and cybersecurity

## Future Improvements
- Deep learning models (LSTM, BERT)
- Real-time social media data
- Web dashboard using Flask
- Multilingual support
  
