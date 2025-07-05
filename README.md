
# 🕵️ Transaction Fraud Detection

This project detects fraudulent transactions using machine learning. It includes data preprocessing, model training (e.g., Random Forest, XGBoost), and deployment via a Streamlit web application for real-time predictions.

---

## 📁 Project Structure

```
Transaction-Fraud-Detection-ML/
├── data/                  # Raw and processed data files
├── backend/
│   ├── main.py            # Main script to run training/prediction
│   ├── model/
│   │   └── prediction_model.py  # ML logic and evaluation
├── streamlit_app.py       # Streamlit frontend
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/Transaction-Fraud-Detection-ML.git
cd Transaction-Fraud-Detection-ML
```

### 2. Create a virtual environment and activate it

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

---

## 🧠 ML Models Used

* Random Forest
* XGBoost

Evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

---

## 📊 Features

* Fraud vs. Non-Fraud classification
* Real-time prediction via UI
* Modular, scalable ML pipeline

---

## 📌 Requirements

* Python 3.8+
* pandas, scikit-learn, xgboost
* streamlit

---

## 📬 Contact

Feel free to reach out via GitHub Issues or Pull Requests if you'd like to contribute or report bugs!

---

Let me know if you'd like to include:

* Dataset source
* Sample predictions
* Screenshots of the UI
