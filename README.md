# Titanic Survival Prediction

**CODSOFT Data Science Internship — Task 1**

## 📌 Problem Statement
Build a machine learning model that predicts whether a Titanic passenger survived, using attributes such as age, sex, ticket class, fare, and cabin.

## 📊 Dataset
[Titanic Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
891 passengers, 12 columns: `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

## 🛠️ Approach
1. **EDA** — visualized survival rate against sex, class, age, fare, and embarkation port.
2. **Feature Engineering**
   - Extracted `Title` (Mr/Mrs/Miss/Master/Rare) from passenger names
   - Created `FamilySize` and `IsAlone` from `SibSp` + `Parch`
   - Created `HasCabin` flag (cabin data is mostly missing, but presence is informative)
   - Imputed missing `Age` using median age per title group
   - Binned `Age` and `Fare` into categories
3. **Modeling** — trained and compared three classifiers:
   - Logistic Regression
   - Decision Tree
   - Random Forest
4. **Evaluation** — accuracy, 5-fold cross-validation, confusion matrix, and feature importance.

## 📈 Results

| Model | Test Accuracy | CV Mean Accuracy |
|---|---|---|
| Logistic Regression | ~81.6% | ~80.2% |
| Decision Tree | ~74.9% | ~80.2% |
| **Random Forest (best)** | **~83.2%** | **~82.5%** |

**Top predictive features:** Sex, Title, Fare, Age, Pclass — consistent with the historical "women and children first" evacuation priority and the survival advantage of wealthier passengers.

## 🌐 Interactive Web Predictor
A standalone web app (`titanic_predictor.html`) runs the trained logistic regression model directly in the browser — no server needed. Enter passenger details and get a live survival prediction with confidence score.

**To use:** just double-click `titanic_predictor.html` to open it in any browser.

## 📁 Repo Structure
```
CODSOFT_TASK1/
├── data/
│   └── titanic.csv
├── images/
│   ├── eda_overview.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── Titanic_Survival_Prediction.ipynb
├── titanic_pipeline.py
├── titanic_predictor.html
└── README.md
```

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Titanic_Survival_Prediction.ipynb
```
or run the standalone script:
```bash
python titanic_pipeline.py
```

## 🔖 Tags
`#codsoft` `#internship` `#datascience` `#machinelearning` `#python`

---
*Task 1 of 3 — CODSOFT Data Science Internship*
