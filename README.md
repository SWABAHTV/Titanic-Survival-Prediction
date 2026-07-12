# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using the classic Kaggle Titanic dataset. This project covers the full workflow: exploratory data analysis, feature engineering, and comparison of two classification models.

## Project Overview

- **Goal:** Predict whether a passenger survived the Titanic disaster based on features like class, sex, age, and family size.
- **Type:** Binary classification
- **Dataset:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) (Kaggle)

## Workflow

1. **Exploratory Data Analysis (EDA)**
   - Checked class balance, survival by sex, survival by passenger class
   - Visualized age distribution and feature correlations with a heatmap
2. **Data Cleaning & Feature Engineering**
   - Filled missing `Age` values with the median
   - Filled missing `Embarked` values with the mode
   - Dropped `Cabin` (too many missing values)
   - Engineered `FamilySize` (SibSp + Parch + 1) and `IsAlone` flag
   - Encoded `Sex` and `Embarked` as numeric
3. **Modeling**
   - Trained a **Logistic Regression** model
   - Trained a **Random Forest Classifier** (200 trees, max depth 5)
4. **Evaluation**
   - Compared accuracy, confusion matrix, and classification report
   - Extracted feature importances from the Random Forest model

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 79.9% |
| Random Forest | 81.6% |

**Top predictive features (Random Forest):** Sex, Fare, Pclass, Age

Random Forest outperformed Logistic Regression, with `Sex` by far the strongest predictor of survival — consistent with the historical "women and children first" evacuation priority.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Project Structure

```
titanic-survival-prediction/
├── titanic_project.ipynb   # Full notebook: EDA, feature engineering, modeling
├── titanic_model.pkl        # Saved (pickled) trained Random Forest model
├── requirements.txt
└── README.md
```

## Author

Muhmmd Sabah
