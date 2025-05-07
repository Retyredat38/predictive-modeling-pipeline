# Airline Ticket Price Prediction – Machine Learning Pipeline

This project develops a machine learning pipeline to predict airline ticket prices using historical flight data. It includes robust data preprocessing, automated model selection using cross-validation, and exports the best-performing model for production use.

## Project Purpose

Airline ticket prices are notoriously variable due to many factors — including route, airline, timing, demand, and more. The goal of this project is to:

- Predict ticket prices based on historical trends
- Evaluate multiple ML models and select the best one
- Provide a modular and reusable training pipeline

This type of analysis could be useful for:

- **Data scientists** exploring regression modeling and AutoML strategies
- **Travel industry analysts** estimating ticket price ranges
- **Educational purposes** to demonstrate real-world model development

---

## Features

- **Data Preprocessing**: Cleans missing values, encodes categorical variables
- **Model Training**: Trains and compares:
  - Random Forest Regressor
  - Ridge Regressor
  - XGBoost Regressor
- **Cross-Validation**: RMSE-based scoring
- **Model Export**: Best model saved to `models/` directory
- **Modular Codebase**: Organized into `src/` for clarity and reusability

---

## Project Structure
ai_automl_project/
├── data/ # Source CSV data (excluded from Git)
├── models/
│ └── XGBoost_model.pkl # Best trained model
├── src/
│ ├── train.py # Main training pipeline
│ └── preprocessing/
│ ├── init.py
│ └── cleaner.py # Data preprocessing logic
├── requirements.txt # Install dependencies
├── .gitignore
└── README.md

## How to Run

1. **Clone the repo**:

```bash
git clone https://github.com/YOUR_USERNAME/ai_automl_project.git
cd ai_automl_project

2. **Install dependencies
pip install -r requirements.txt

3. **Place your CSV in the data/ folder
    The project expects a file named: data/Airline_Ticket_Price_data.csv

4. **Run the training pipeline
    python -m src.train

5. **The best model will be saved to:
    modes/XGBoost_model.pkl

** Model Evaluation Results
Below are sample results from a recent training run (using CPU):

Model	CV RMSE (↓ better)
Random Forest	43,144,735
Ridge	98,979,700
XGBoost	33,477,038

XGBoost was selected as the best model and saved.

** **Customization Tips
To change which models are trained, modify the train_models() function in src/train.py.

If your CSV has different columns, adjust preprocess_data() in src/preprocessing/cleaner.py.

# For GPU-accelerated XGBoost, install with:


pip install xgboost[scikit-learn]
📄 Requirements

Key packages:


pandas
scikit-learn
xgboost
optuna
joblib

# Install all via:


pip install -r requirements.txt
⚖️ License
MIT License – free to use, modify, and distribute.

**Author
Developed by Me – 2025

Proudly trained and tested on local CPU hardware.
Feedback and forks welcome on GitHub!

