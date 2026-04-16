from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model = pickle.load(open("movie_model.pkl", "rb"))
vectorizer = pickle.load(open("movie_vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned = clean_text(review)
    vect = vectorizer.transform([cleaned])

    prediction = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0].max()

    if prediction == 1:
        result = f"Positive 😊 (Confidence: {prob:.2f})"
    else:
        result = f"Negative 😡 (Confidence: {prob:.2f})"

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
