# 🎬 Movie Review Sentiment Analyzer

A machine learning web app that analyzes movie reviews and classifies them as **positive** or **negative** — built with Python and deployed on Streamlit.

🔗 **Live Demo:** [Click here](https://sentiment-analysis-app-zmf6pikddbrqyqshgxcpdh.streamlit.app/) <!-- replace with your actual link -->

---

## 🖼️ Preview

> <img width="1490" height="961" alt="Screenshot from 2026-03-18 21-23-34" src="https://github.com/user-attachments/assets/f163bfe8-3090-4feb-b996-77344222fce2" />


## 🛠️ Built With

- **Python**
- **Scikit-learn** — Logistic Regression model
- **TF-IDF Vectorizer** — text feature extraction
- **Streamlit** — Web app interface
- **Pickle** — Model serialization

---

## 🚀 How It Works

1. User enters a movie review (or tries a random sample)
2. Text is cleaned — lowercased, HTML tags and punctuation removed
3. Review is vectorized using a trained **TF-IDF Vectorizer**
4. **Logistic Regression** model predicts sentiment
5. Result is displayed with a **confidence percentage**

---

## 🧠 Model Details

| Detail | Value |
|---|---|
| Dataset | [IMDB Dataset of 50K Movie Reviews ](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| Algorithm | Logistic Regression |
| Vectorizer | TF-IDF (max 10,000 features) |
| Accuracy | ~89% on test set |
| Task | Binary classification (positive / negative) |

### Why Logistic Regression over Naive Bayes?
Both models were trained and evaluated. Logistic Regression outperformed Multinomial Naive Bayes on this dataset, so it was selected as the final model.

---

## 📁 Project Structure

```
sentiment-analysis/
├── app.py                            # Streamlit web app
├── Sentiment-Analysis.py             # Model training script
├── model.pkl                         # Trained Logistic Regression model
├── vectorizer.pkl                    # Fitted TF-IDF vectorizer
├── requirements.txt                  # Dependencies
└── README.md
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/Shreeya788/Sentiment-Analysis-App.git
cd sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Train the model first
python train.py

# Run the app
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit==1.43.0
scikit-learn
pandas
numpy
```

