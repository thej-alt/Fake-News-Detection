# 📰 Fake News Detection using NLP & Logistic Regression

This project builds a **machine learning model** to classify news articles as **Fake** or **Real** using **TF-IDF Vectorization** and **Logistic Regression**, with an ensemble approach combining **Logistic Regression** and **Multinomial Naive Bayes**.

---

## 📌 Workflow

1. **Load Dataset** – Import and inspect news data.  
2. **Preprocess & Split Data** – Clean text, handle missing values, and split into training and test sets.  
3. **Vectorize with TF-IDF** – Convert text data into TF-IDF feature vectors.  
4. **Train Models**  
   - Logistic Regression (`max_iter=1000`, `class_weight="balanced"`)  
   - Multinomial Naive Bayes  
5. **Ensemble Predictions** – Weighted average: `0.6*LR + 0.4*NB`  
6. **Evaluate Models**  
   - Accuracy Score  
   - Classification Report  
   - Confusion Matrix  
7. **Predict Custom Inputs** – Generate predictions with confidence scores and thresholds.

---

## ⚙️ Ensemble Model Performance

- **Ensemble Accuracy:** 97.51%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fake  | 0.99      | 0.97   | 0.98     | 4696    |
| Real  | 0.96      | 0.98   | 0.97     | 4284    |
| **Accuracy** | - | - | 0.98 | 8980 |
| **Macro Avg** | 0.97 | 0.98 | 0.98 | 8980 |
| **Weighted Avg** | 0.98 | 0.98 | 0.98 | 8980 |

**Confusion Matrix:**  
*(Displays True vs Predicted Labels for Ensemble Model)*

