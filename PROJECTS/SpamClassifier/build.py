# build_model.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# Download NLTK resources
#nltk.download("punkt")
#nltk.download("stopwords")

# Text transformation function
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    cleaned = [token for token in tokens if token.isalnum()]
    filtered = [token for token in cleaned if token not in stopwords.words("english")]
    stemmed = [ps.stem(token) for token in filtered]
    return " ".join(stemmed)

# Load dataset
df = pd.read_csv("D:\ML\CODES\Projects\SpamNotSpam\spam.csv", encoding="latin-1")

# Keep relevant columns
df = df[['v1', 'v2']]
df = df.rename(columns={'v1': 'target', 'v2': 'text'})

# Encode labels
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Drop duplicates
df.drop_duplicates(inplace=True)

# --- EDA (Save plots instead of printing) ---
sns.countplot(x='target', data=df)
plt.title("Class Distribution")
plt.savefig("class_distribution.png")
plt.clf()

df['num_characters'] = df['text'].apply(len)
sns.histplot(df[df['target']==0]['num_characters'], color='green', label='Ham', kde=True)
sns.histplot(df[df['target']==1]['num_characters'], color='red', label='Spam', kde=True)
plt.legend()
plt.title("Message Length Distribution")
plt.savefig("length_distribution.png")
plt.clf()

# --- Preprocessing ---
df['transformed_text'] = df['text'].apply(transform_text)

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Save metrics to a file
with open("model_metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))

# Save model and vectorizer
pickle.dump(cv, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model and vectorizer saved. Metrics and plots generated.")
