# 🛡️ Intrusion Detection System (IDS) with ML Ensemble + Streamlit Dashboard

An end-to-end Machine Learning pipeline for network intrusion detection, trained on the **X-IIoTID dataset**, featuring a real-time Streamlit dashboard for live traffic simulation and threat visualization.

---

## 📌 Project Overview

This project builds a robust IDS using an ensemble of gradient boosting models (XGBoost + LightGBM), with:
- **CTGAN-based data balancing** for imbalanced attack classes
- **Bayesian Hyperparameter Optimization** for model tuning
- **Random Forest feature selection** to reduce dimensionality
- **Streamlit dashboard** for real-time intrusion monitoring and visualization

---

## 🗂️ Project Structure

```
ids-project/
│
├── new_code_1__1___1_.ipynb   # Full ML training pipeline (Colab notebook)
├── ids_dashboard.py           # Streamlit dashboard app
│
├── ids_ensemble_model.pkl     # Trained ensemble model (generated after training)
├── label_encoder.pkl          # Label encoder (generated after training)
├── selected_features.pkl      # Selected feature list (generated after training)
├── scaler.pkl                 # Feature scaler (generated after training)
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

> ⚠️ The `.pkl` model files are **not included** in this repo (too large). Run the notebook to generate them, or download from the release section.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ids-project.git
cd ids-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

Open and run `new_code_1__1___1_.ipynb` in Google Colab (recommended) or Jupyter.  
It will generate the required `.pkl` files.

### 4. Run the Dashboard

```bash
streamlit run ids_dashboard.py
```

---

## 📊 Dashboard Features

| Feature | Description |
|---|---|
| 🔴 Real-time simulation | Simulates network traffic flows and classifies them live |
| 📊 Attack breakdown chart | Bar chart of detected attack types |
| 🟢 Normal vs Attack count | Live counter updated per flow |
| 🎯 Accuracy tracking | Compares model prediction vs simulated label |
| 📋 Flow log table | Last 10 predictions with confidence scores |

---

## 🧠 ML Pipeline

```
Raw CSV (X-IIoTID)
    ↓
Data Cleaning & Label Encoding
    ↓
Train/Test Split
    ↓
RF-based Feature Selection
    ↓
CTGAN Synthetic Balancing
    ↓
Bayesian Optimization (XGBoost + LightGBM)
    ↓
Ensemble Model Training
    ↓
Evaluation & Model Serialization (.pkl)
```

---

## 📦 Dataset

This project uses the [X-IIoTID dataset](https://ieee-dataport.org/documents/x-iiotid-connectivity-agnostic-and-target-independent-intrusion-dataset-industrial-internet).  
Download and place it in your Google Drive at the path configured in the notebook (`CFG['path']`).

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Scikit-learn** — preprocessing, feature selection, evaluation
- **XGBoost / LightGBM** — ensemble classifiers
- **CTGAN** — synthetic data generation for class balancing
- **BayesianOptimization** — hyperparameter tuning
- **Streamlit** — interactive dashboard
- **Matplotlib / Seaborn** — visualizations
- **Joblib** — model serialization

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
