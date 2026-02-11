# Import necessary lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

## Download NLTK resources ##
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

df = pd.read_csv('wk08/twitter_training.csv')
print(df.head())

# Display dataset info
print("Dataset Shape:", df.shape)
print("\nMissing Values Before Cleaning:")
print(df.isna().sum())

## Task 1: Data Cleaning and Preprocessing
def clean_text(text):
    if pd.isna(text):
        return ""
    try:
        text = str(text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    try:
        text = clean_text(text)
        if not text.strip():
            return ""
        # Tokenize and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

# Apply preprocessing with progress feedback
print("\nPreprocessing text data...")
df['cleaned_text'] = df['Tweet content'].apply(preprocess_text)

# Handle missing values after cleaning
initial_count = len(df)
df = df[df['cleaned_text'].str.strip() != '']
df = df.dropna(subset=['sentiment'])
print(f"\nRemoved {initial_count - len(df)} rows with empty/missing data")

## Task 2: Data Splitting
x = df['cleaned_text']
y = df['sentiment'].astype(str)     # Ensure sentiment is string type
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

## Task 3: Feature Extraction (TF-IDF)
print("\nCreating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),         # Include bigrams
    min_df=5,                   # Ignore terms that appear in fewer than 5 documents
    max_df=0.7                  # Ignore terms that appear in more than 70% of documents
)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

## Task 4: Model Building (Multinomial Naive Bayes)
print("\nTraining model...")
nb_classifier = MultinomialNB(alpha=0.1)    # Add smoothing
nb_classifier.fit(x_train_tfidf, y_train)

## Task 5: Model Evaluation
y_pred = nb_classifier.predict(x_test_tfidf)

print("\nModel Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Confusion matrix heatmap
plt.plot(1, 2, 2)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=nb_classifier.classes_,
            yticklabels=nb_classifier.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Sentiment distribution
plt.plot(1, 2, 1)
sns.countplot(x='sentiment', data=df, order=df['sentiment'].value_counts().index)
plt.title('Distribution of Sentiments')
plt.xticks(rotation=45)

## Task 6: Prediction with error handling
sample_tweets = [
    "I will finish Borderlands 2 today. I have some new commands set up and am looking forward to a good stream! Start in about an hour!",
    "Now how do I submit any complaint? Your CEO isn't offering his staff their bonuses.",
    "All the Borderlands are damn rubbish"
]

print("\nSample Predictions:")
for tweet in sample_tweets:
    try:
        cleaned = preprocess_text(tweet)
        if not cleaned.strip():
            print(f"Could not process tweet: '{tweet}'")
            continue
        vec = tfidf_vectorizer.transform([cleaned])
        pred = nb_classifier.predict(vec)[0]
        proba = nb_classifier.predict_proba(vec).max()
        print(f"'{tweet[:50]}...' -> {pred} (confidence: {proba:.2f})")
    except Exception as e:
        print(f"Error predicting for tweet: {e}")

# ----------------------------------------------------------------- #

# Import required libraries
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification

# Create pipeline for sentiment analysis
classification = pipeline('sentiment-analysis')
type(classification)

# Read data into DataFrame
df = pd.read_csv("wk08/twitter_training.csv")
# Set the columns
df.columns = ['id','entity','sentiment','Tweet content']
# Drop null values
df = df.dropna()
# Print first 5 rows
print(df.head())

# Select only required columns
df = df[['sentiment','Tweet content']]

# Let’s check the number of classes available in dataset and their distribution using histogram.
df['sentiment'].hist()
target_map = { 'Positive': 1, 'Negative': 0}
df['target'] = df['sentiment'].map(target_map)
texts = df['Tweet content'].to_list()

import torch
torch.cuda.is_available()

# Print predictions
predictions = classification(texts)
print(predictions[:10])

# Print few probabilities
probs = [d['score'] if d['label'].startswith('P') else 1 - d['score'] for d in predictions ]
print(probs[:10])

preds = [1 if d['label'].startswith('P') else 0 for d in predictions]

# Convert into numpy array 
preds = np.array(preds)
print("acc:", np.mean(df['target'] == preds))

# Calculate confusion matrix
cm = confusion_matrix(df['target'], preds)

# Create function for plotting confusion matrix
def plot_cm(cm):
  classes = ['negative','positive']
  df_cm = pd.DataFrame(cm, index=classes, columns=classes)
  ax = sns.heatmap(df_cm, annot = True, fmt='g')
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')
  plt.show()

plot_cm(cm)