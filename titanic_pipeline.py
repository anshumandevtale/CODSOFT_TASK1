import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_style('whitegrid')

# 1. Load data
df = pd.read_csv('data/titanic.csv')
print("Shape:", df.shape)
print(df.info())

# 2. EDA
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
sns.countplot(x='Survived', data=df, ax=axes[0,0]).set_title('Survival Count')
sns.countplot(x='Pclass', hue='Survived', data=df, ax=axes[0,1]).set_title('Survival by Pclass')
sns.countplot(x='Sex', hue='Survived', data=df, ax=axes[0,2]).set_title('Survival by Sex')
sns.histplot(data=df, x='Age', hue='Survived', kde=True, ax=axes[1,0], bins=30).set_title('Survival by Age')
sns.countplot(x='Embarked', hue='Survived', data=df, ax=axes[1,1]).set_title('Survival by Embarked')
sns.boxplot(x='Survived', y='Fare', data=df, ax=axes[1,2]).set_title('Fare vs Survival')
plt.tight_layout()
plt.savefig('images/eda_overview.png', dpi=120)
plt.close()

# 3. Feature engineering
df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\.')
title_map = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
    'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
    'Capt': 'Rare', 'Sir': 'Rare'
}
df['Title'] = df['Title'].map(title_map).fillna('Rare')

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['HasCabin'] = df['Cabin'].notna().astype(int)

# Impute Age by median per Title group
df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100],
                       labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Mid', 'High', 'VeryHigh'])

# 4. Encode categoricals
le = LabelEncoder()
for col in ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin']:
    df[col] = le.fit_transform(df[col].astype(str))

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Title', 'FamilySize', 'IsAlone', 'HasCabin', 'AgeBin', 'FareBin']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Train + compare models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(model, X, y, cv=5)
    results[name] = {'test_acc': acc, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
    print(f"\n{name}: Test Acc={acc:.4f}, CV Mean={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(classification_report(y_test, preds))

# 6. Best model - confusion matrix + feature importance
best_name = max(results, key=lambda k: results[k]['test_acc'])
best_model = models[best_name]
print(f"\nBest model: {best_name}")

preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Died','Survived'], yticklabels=['Died','Survived'])
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=120)
plt.close()

if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f'Feature Importance - {best_name}')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png', dpi=120)
    plt.close()
    print("\nFeature Importances:\n", importances)

print("\n=== SUMMARY ===")
for name, r in results.items():
    print(f"{name}: Test Acc={r['test_acc']:.4f}, CV Mean={r['cv_mean']:.4f}")
