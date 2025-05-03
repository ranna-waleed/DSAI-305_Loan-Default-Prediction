# DSAI-305_Loan-Default-Prediction

# Loan Default Prediction:
Loan default prediction needs to be precise because it protects the business interests of lenders as well as borrowers. The prediction technique assists lenders to avoid financial damage through its ability to detect high-risk loans before they grant approval. The system leads to responsible credit assessment methods for borrowers which improves the conditions they receive through loans.

An investigation of numerous machine learning methods and interpretability approaches exists to predict loan defaults with special consideration for an unbalanced dataset distribution.

# Dataset
Source: Lending Club Loan Data: https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv?select=loan.csv
Size: ~1.19 GB
Format: CSV
Rows: ~890,000
Columns: 77 features
The prediction accuracy faces challenges because the dataset contains far more instances of non-defaults than defaults.

# Models Implemented
Training sessions for every model are present within single dedicated Notebook files which serve both for training and evaluation purposes along with explanation features.

- Random Forest
- Extra Trees Classifier
- Gaussian Naive Bayes
- Logistic Regression
- TabNet
- XGBoost
- Decision Tree
- LightGBM
- Random Forest with SMOTE 
- Deep Neural Network (DNN)

# Explainability Techniques Used
Interpretation and explainability tools help users view the effects of features and monitor model behavior in addition to each model implementation.

- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Partial Dependence Plots (PDP)
- Individual Conditional Expectation (ICE)
- Permutation Feature Importance (PFI)
- Global Surrogate Models

# Feature Engineering & Preprocessing
The separate notebook contains instructions for the following tasks:

- Exploratory Data Analysis (EDA)
- Handling missing values
- Categorical encoding
- Feature scaling
- Outlier treatment
- Imbalanced data handling (e.g., SMOTE)


# Installation & Running Instructions
1. Clone the Repository
git clone https://github.com/yourusername/loan-default-prediction.git
cd loan-default-prediction

2. Set Up a Python Environment
We recommend using Python 3.8+ and setting up a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Required Libraries
pip install -r requirements.txt
Note: If requirements.txt is not provided, install the key libraries manually:
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn shap lime tabnet

4. Download the Dataset
Download Lending Club Loan Data from the official source or provided link (https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv?select=loan.csv)

5. Run Notebooks
