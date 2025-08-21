import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- 1. Load dataset ---
DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(DATA_PATH)

# --- 2. Preprocess ---
# Replace blank strings with NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(0, inplace=True)

X = df.drop(['Churn'], axis=1)
y = df['Churn'].map({'No': 0, 'Yes': 1})

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

categorical_processor = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
numeric_processor = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_processor, categorical_cols),
        ('num', numeric_processor, numeric_cols)
    ],
    remainder='drop'
)

# --- 3. Build pipeline ---
pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])

# --- 4. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Hyperparameter tuning ---
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# --- 6. Evaluate ---
print("Best parameters:", grid.best_params_)
y_pred = grid.predict(X_test)
print("Model performance:")
print(classification_report(y_test, y_pred))

# --- 7. Save best model ---
best_pipe = grid.best_estimator_
joblib.dump(best_pipe, os.path.join(os.getcwd(), "best_pipe.pkl"))
print("Pipeline saved at", os.path.join(os.getcwd(), "best_pipe.pkl"))
