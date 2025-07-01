# 📩 Real-Time Spam Detection Web App using Machine Learning

This is a real-time spam classifier that detects whether an input message is **SPAM** or **NOT SPAM** using a machine learning model trained on the SMS Spam Collection dataset. The web interface is built using **Streamlit** for simplicity and ease of use.

---

## 🚀 Features

- 🔍 Real-time spam detection via text input
- ✅ Trained on 5,000+ labeled SMS messages
- 📊 Accuracy: ~96.6%
- 🧠 Machine Learning: TF-IDF + Multinomial Naive Bayes
- 🌐 Web interface using Streamlit

---

## 📁 Project Structure
   spam-detector-app/
├── app.py # Streamlit frontend
├── model_training.py # Training script
├── requirements.txt # Required libraries
├── model/
│ └── spam_classifier.pkl # Saved ML model
├── data/
│ └── spam.csv # Dataset (SMS Spam Collection)
├── venv/ # Virtual environment


---

## ⚙️ Setup Instructions

### 1. ✅ Clone the Repository

'bash
> git clone https://github.com/Sanjay767676/spam-detector-app.git
> cd spam-detector-app


### 2. 🐍 Create a Virtual Environment (Python 3.10+)
 > py -3.10 -m venv venv
> .\venv\Scripts\activate

### If PowerShell blocks activation, run:
  >  Set-ExecutionPolicy RemoteSigned

Then re-run:
   > .\venv\Scripts\activate

### 3. 📦 Install Required Libraries
   >  pip install -r requirements.txt

### 4. 📊 Train the Model
   > python model_training.py

***This will:
Load and clean the dataset
Train the ML model
Save it as model/spam_classifier.pkl***

### 5. 🌐 Run the Web App
streamlit run app.py
Your app will be available at:
👉 http://localhost:8501

