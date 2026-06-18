# Customer Churn Prediction

Binary classification model to predict customer churn using the Telco Customer Churn dataset. Built during Internspark AI internship.

## Overview
- **Problem:** Predict whether a customer will churn based on demographics, account info, and service usage.
- **Dataset:** IBM Telco Customer Churn (7,043 rows, 21 features).
- **Target:** Churn (Yes/No) — imbalanced (~27% churn).

## Approach
1. EDA: Class balance check, churn by contract type, numeric feature distributions.
2. Preprocessing: Label encoding, StandardScaler, stratified train/test split.
3. Models:
   - Logistic Regression (baseline)
   - Random Forest (100 trees, max_depth=10)
4. Cross-validation: 5-fold stratified, scoring = ROC-AUC.
5. Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, ROC curves.

## Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.7991 | 0.6426 | 0.5481 | 0.5916 | 0.8402 |
| Random Forest | 0.7956 | 0.6463 | 0.5080 | 0.5689 | 0.8358 |

## Key Features
- Tenure, Contract type, Monthly Charges, Total Charges, Internet Service.

## Files
- `Task1_Churn_Classification.ipynb` — Full notebook.
- Model artifacts (Google Drive): churn_model_rf.pkl, scaler.pkl, feature_names.pkl.

## Tools
Python, Scikit-learn, Pandas, Matplotlib, Seaborn, Google Colab

## Author
[Alan Raju](https://www.linkedin.com/in/alan-raju-09bb0040a/) | [GitHub](https://github.com/zom1xo)
