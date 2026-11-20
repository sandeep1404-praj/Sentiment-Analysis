"""
Flask Backend API for Hindi Sentiment Analysis
Loads the trained SVM model and TF-IDF vectorizer saved by sentiment.ipynb
"""

import re
import os
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

MODEL_PATH = "hindi_sentiment_svm.pkl"
TFIDF_PATH = "hindi_sentiment_tfidf.pkl"

model = None
tfidf = None

def load_model():
    global model, tfidf
    if os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH):
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        print("[INFO] Model and vectorizer loaded successfully.")
        return True
    else:
        print("[WARNING] Model files not found. Please run the notebook first.")
        return False

load_model()


# ============================================================
# TEXT PREPROCESSING (same as notebook)
# ============================================================

def clean_hindi_text(text):
    text = str(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    # Keep Hindi Unicode characters, numbers and spaces
    text = re.sub(r"[^\u0900-\u097F\s0-9]", " ", text)
    # Remove extra spaces again
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# SENTIMENT LABELS
# ============================================================

sentiment_names = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

sentiment_emojis = {
    "Negative": "😞",
    "Neutral":  "😐",
    "Positive": "😊"
}

sentiment_colors = {
    "Negative": "#ff4757",
    "Neutral":  "#ffa502",
    "Positive": "#2ed573"
}


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or tfidf is None:
        # Try reloading
        if not load_model():
            return jsonify({
                "error": "Model not found. Please run the Jupyter notebook (sentiment.ipynb) first to train and save the model."
            }), 503

    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleaned = clean_hindi_text(text)

    if not cleaned:
        return jsonify({"error": "Text contains no valid Hindi characters. Please enter Hindi text."}), 400

    vector = tfidf.transform([cleaned])
    prediction_id = int(model.predict(vector)[0])
    label = sentiment_names[prediction_id]

    # Get confidence scores if model supports it
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        confidence = {
            sentiment_names[i]: round(float(p) * 100, 2)
            for i, p in enumerate(proba)
        }
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vector)[0]
        # Normalize decision scores to rough confidence
        if hasattr(scores, "__len__"):
            exp_scores = [max(0, float(s)) for s in scores]
            total = sum(exp_scores) + 1e-9
            confidence = {
                sentiment_names[i]: round(exp_scores[i] / total * 100, 2)
                for i in range(len(exp_scores))
            }

    return jsonify({
        "text": text,
        "cleaned_text": cleaned,
        "sentiment": label,
        "sentiment_id": prediction_id,
        "emoji": sentiment_emojis[label],
        "color": sentiment_colors[label],
        "confidence": confidence
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "model_loaded": model is not None and tfidf is not None,
        "model_path": MODEL_PATH,
        "tfidf_path": TFIDF_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "tfidf_exists": os.path.exists(TFIDF_PATH),
    })


@app.route("/batch", methods=["POST"])
def batch_predict():
    if model is None or tfidf is None:
        if not load_model():
            return jsonify({"error": "Model not found. Run the notebook first."}), 503

    data = request.get_json(force=True)
    texts = data.get("texts", [])

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Provide a list of texts under 'texts' key."}), 400

    results = []
    for text in texts[:20]:  # Cap at 20 to avoid overload
        cleaned = clean_hindi_text(str(text))
        if not cleaned:
            results.append({"text": text, "error": "No valid Hindi text"})
            continue
        vector = tfidf.transform([cleaned])
        pred_id = int(model.predict(vector)[0])
        label = sentiment_names[pred_id]
        results.append({
            "text": text,
            "sentiment": label,
            "emoji": sentiment_emojis[label],
            "color": sentiment_colors[label]
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    print("=" * 60)
    print("  Hindi Sentiment Analysis API")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, port=5000)
