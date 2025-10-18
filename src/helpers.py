import joblib
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from config import *

# Tune Model
def tune_model(pipe: Pipeline, params: dict, X_train, Y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid  = GridSearchCV(pipe, params, cv=skf, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, Y_train)
    best_model = grid.best_estimator_
    
    return best_model

# Save Model
def save_model(model ,file_name:str):
    os.makedirs(name='../models', exist_ok=True)
    joblib.dump(model,f'../models/{file_name}.pkl')
    
# Evaluating Metrics
def evaluate_model(model_name:str, scaler_name:str, model, X_test, Y_test) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        'model_name': model_name,
        'scaler_name': scaler_name,
        'accuracy': accuracy_score(Y_test, y_pred),
        'f1_score': f1_score(Y_test, y_pred, average='weighted'),
        'precision': precision_score(Y_test, y_pred, average='weighted'),
        'recall': recall_score(Y_test, y_pred, average='weighted'),
    }
    return metrics