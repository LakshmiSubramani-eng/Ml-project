movie_reviews = [
    # POSITIVE
    "The movie was fantastic and full of entertainment",
    "Absolutely loved the film and the acting was great",
    "Amazing direction and wonderful music",
    "Brilliant movie must watch",
    "Excellent visuals and great storyline",
    "Outstanding performance by the lead actor",
    "A beautiful movie with a powerful message",
    "One of the best movies I have seen",

    # NEGATIVE
    "Worst movie ever and complete waste of time",
    "The storyline was boring and too slow",
    "Poor script and very bad acting",
    "Terrible movie with no logic",
    "Not good and very disappointing",
    "Bad direction and weak screenplay",
    "Awful movie I regret watching it",
    "Extremely boring and badly executed film"
]
import re
import pickle
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('vader_lexicon')

# ----------------------------
# Dataset
# ----------------------------
df = pd.DataFrame({'review': movie_reviews})

# ----------------------------
# Auto-label using STRICT threshold
# ----------------------------
sia = SentimentIntensityAnalyzer()

def auto_label(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 1
    elif score <= -0.05:
        return 0
    else:
        return None   # remove neutral

df['sentiment'] = df['review'].apply(auto_label)
df = df.dropna()  # remove neutral samples

# ----------------------------
# Text Cleaning
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

# ----------------------------
# Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# ----------------------------
# Train Model
# ----------------------------
model = LogisticRegression(class_weight='balanced')
model.fit(X, y)

# ----------------------------
# Save Model
# ----------------------------
pickle.dump(model, open("movie_model.pkl", "wb"))
pickle.dump(vectorizer, open("movie_vectorizer.pkl", "wb"))

print("Balanced model trained and saved successfully")
