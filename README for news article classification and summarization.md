📰 News Article Summarization and Sentiment Classification
This project performs web scraping, summarization using GPT-2, sentiment analysis with VADER, and sentiment classification using a Random Forest classifier.

📁 Project Structure
bash
Copy
Edit
newsarticle.ipynb         # Main Jupyter Notebook
random_forest_model.pkl   # Trained Random Forest Classifier
tfidf_vectorizer.pkl      # Fitted TF-IDF Vectorizer
🚀 Features
Scrapes latest news headlines and articles from The Star

Preprocesses and cleans article content

Summarizes news articles using the GPT-2 model

Computes ROUGE scores to evaluate summary quality

Performs sentiment analysis using VADER

Balances dataset and trains a Random Forest classifier to predict sentiment

Saves the trained model and vectorizer

📦 Dependencies
Install required packages using the following commands:

bash
Copy
Edit
pip install textblob
pip install vaderSentiment
pip install rouge-score
pip install torch==2.0.0 torchvision torchaudio
pip install transformers==4.30.0
pip install textblob==3.0.0
pip install spacytextblob==0.1.7
python -m spacy download en_core_web_sm
pip install imbalanced-learn
Also ensure these Python libraries are imported:

python
Copy
Edit
requests, bs4, pandas, re, nltk, sklearn, matplotlib, seaborn, joblib
🧠 Model Workflow
1. Web Scraping
Scrapes article headlines and links from 25 pages.

Extracts full text content for each article.

2. Preprocessing
Removes special characters and normalizes spacing.

Prepares content for summarization and sentiment analysis.

3. Summarization with GPT-2
Loads GPT-2 from Hugging Face Transformers.

Generates summaries for each article using generate().

4. ROUGE Score Evaluation
Compares generated summary with extractive (first 3 sentences).

Computes ROUGE-1, ROUGE-2, and ROUGE-L scores.

5. Sentiment Analysis with VADER
Assigns sentiment labels: Positive, Neutral, Negative.

6. Classification with Random Forest
Uses TF-IDF for feature extraction.

Balances the dataset using oversampling.

Splits data, trains, and evaluates a Random Forest classifier.

Outputs accuracy and confusion matrix.

7. Model Export
Saves the trained RandomForestClassifier and TfidfVectorizer using joblib.

📊 Results
Accuracy scores for training, testing, and overall dataset.

ROUGE Scores evaluating summarization quality.

Confusion Matrix visualizing classifier performance.

Sentiment Distribution before and after balancing.

🧪 Usage
You can reuse the trained model for inference:

python
Copy
Edit
import joblib

# Load model and vectorizer
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Predict new article sentiment
new_article = "The government has announced new economic reforms."
X_new = vectorizer.transform([new_article])
prediction = model.predict(X_new)
print(prediction)
🗂️ Notes
Make sure the NLTK resources like 'wordnet' and 'punkt' are downloaded using:

python
Copy
Edit
import nltk
nltk.download('wordnet')
nltk.download('punkt')
Adapt scraping logic if the website layout changes.

GPT-2 is not fine-tuned for summarization — consider T5, Bart, or fine-tuned GPT models for better results.

