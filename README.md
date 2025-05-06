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

- Random Forest, Paper:https://ieeexplore.ieee.org/document/9336801
- Extra Trees Classifier, Paper:https://doi.org/10.1016/j.ijcce.2023.09.001
- Gaussian Naive Bayes, Paper:https://sci-hub.se/https://ieeexplore.ieee.org/abstract/document/7877200
- Logistic Regression , Paper:https://eprints.lse.ac.uk/116375/1/rsos.191649.pdf?form=MG0AV3
- TabNet, Paper:https://www.politesi.polimi.it/retrieve/275d883a-78a3-4206-a3a4-ae0a93d9a41a/Master_Thesis_Arnaldo_Mollo.pdf
- XGBoost, Paper:  https://arxiv.org/pdf/2012.03749
- Decision Tree , Paper:https://www.mdpi.com/2227-7390/12/21/3423
- LightGBM , Paper:https://bcpublication.org/index.php/BM/article/view/1857
- Random Forest with SMOTE , Paper: https://www.researchgate.net/publication/338286615_A_study_on_predicting_loan_default_based_on_the_random_forest_algorithm
- Deep Neural Network (DNN) , paper:https://www.itm-conferences.org/articles/itmconf/pdf/2025/01/itmconf_dai2024_01012.pdf
- Random Forest ,Paper:https://scispace.com/papers/loan-default-prediction-a-complete-revision-of-lendingclub-2kymtgz0 
- Logistic Regression With Bagging , paper:https://scispace.com/papers/machine-learning-based-loan-default-prediction-in-peer-to-a7r7qmgcd0

# Explainability Techniques Used
Interpretation and explainability tools help users view the effects of features and monitor model behavior in addition to each model implementation.

- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Partial Dependence Plots (PDP)
- Individual Conditional Expectation (ICE)
- Permutation Feature Importance (PFI)
- Global Surrogate Models
- Leave one Feature out (LOFO)
- H-statistics

# Feature Engineering & Preprocessing
The separate notebook contains instructions for the following tasks:

- Exploratory Data Analysis (EDA)
- Handling missing values
- Categorical encoding
- Feature scaling
- Outlier treatment
- Imbalanced data handling (SMOTE)


# Installation & Running Instructions
1. Clone the Repository
git clone https:https://github.com/ranna-waleed/DSAI-305_Loan-Default-Prediction
cd loan-default-prediction

3. Set Up a Python Environment
We recommend using Python 3.8+ and setting up a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install Required Libraries
pip install -r requirements.txt
Note: If requirements.txt is not provided, install the key libraries manually:
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn shap lime tabnet

5. Download the Dataset
Download Lending Club Loan Data from the official source or provided link (https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv?select=loan.csv)

6. Run Notebooks
