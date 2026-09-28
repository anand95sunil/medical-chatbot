import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# 1. Load the training data
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "intents.json"

with open(DATA_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)


# --------------------------------------------------
# 2. Prepare sentences and labels
# --------------------------------------------------

sentences = []
labels = []
responses = {}

for intent in data["intents"]:
    tag = intent["tag"]

    responses[tag] = intent["responses"]

    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(tag)


# --------------------------------------------------
# 3. Convert text into numbers using TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(sentences)


# --------------------------------------------------
# 4. Train the NLU model
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X, labels)


# --------------------------------------------------
# 5. Save the trained model
# --------------------------------------------------

model_data = {
    "model": model,
    "vectorizer": vectorizer,
    "responses": responses
}

MODEL_FILE = BASE_DIR / "model.pkl"

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model_data, file)


print("===================================")
print("Medical Chatbot Model Trained!")
print("===================================")
print(f"Training examples: {len(sentences)}")
print(f"Number of intents: {len(set(labels))}")
print(f"Model saved to: {MODEL_FILE}")