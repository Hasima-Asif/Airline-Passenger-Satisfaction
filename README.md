# 🛫 Airline Passenger Satisfaction Prediction App

Welcome aboard! This interactive app predicts whether a passenger is **satisfied or dissatisfied** with their airline experience — based on their flight details and ratings.

Built as part of my **Introduction to Data Science** project, this Streamlit app wraps up the full machine learning pipeline: from **EDA** to **model training**, to real-time predictions from user input.

---

## 🔍 What the App Does

- ✅ Lets users input flight data and in-flight service ratings
- ✅ Predicts if the passenger is likely **satisfied** or **dissatisfied**
- ✅ Shows clean visualizations and insights from the dataset
- ✅ Explains the model's performance using accuracy and classification report

---

## 🧠 Machine Learning Model

- **Model Used:** Random Forest Classifier 🌲  
- **Training Set Size:** 80% of dataset  
- **Target Variable:** `satisfaction`  
- **Performance Metrics:** Accuracy, F1-score, Confusion Matrix

---

## 📊 Features Used for Prediction

Includes both demographic info and service quality ratings:

- Gender, Age, Customer Type
- Type of Travel (Business or Personal)
- Class (Eco, Eco Plus, Business)
- Flight Distance
- Departure & Arrival Delay (in minutes)
- Inflight Wifi, Food, Seat Comfort, etc. (ratings from 0 to 5)

---

## 🧰 Tech Stack

- `Python`
- `Pandas`, `Seaborn`, `Matplotlib`, `Scikit-learn`
- `Streamlit` for building the web app
- `Pickle` for saving the model and scaler

---

