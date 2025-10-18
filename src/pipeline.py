from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from config import *

# Model Definitions
MODELS = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
        'params': LG_PARAMS
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE),
        'params': RF_PARAMS
    },
    'SVC': {
        'model': SVC(random_state=RANDOM_STATE),
        'params': SVC_PARAMS
    }
}

# Scaler Definitions
SCALERS = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

# Creating Pipelines
def create_pipeline(model_name: str, scaler_name: str) -> Pipeline:
    if model_name not in MODELS.keys() and scaler_name not in SCALERS.keys():
        raise ValueError("Invaid Model or Scaler Name provided")
    else:
        model = MODELS[model_name]['model']
        scaler = SCALERS[scaler_name]
        pipe = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        return pipe