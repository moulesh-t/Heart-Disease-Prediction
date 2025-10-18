import pandas as pd
from sklearn.model_selection import train_test_split
from config import *
from pipeline import MODELS, SCALERS, create_pipeline
from helpers import tune_model, save_model, evaluate_model

# Training Models
def train():
    data = pd.read_csv('./data/heart.csv')
    X = data.drop(['target'], axis=1)
    Y = data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y, random_state=RANDOM_STATE)
    
    results=[]
    
    for model_name,model_def in MODELS.items():
        best_model = None
        best_score = 0
        for scaler_name in SCALERS.keys():
            pipe = create_pipeline(model_name, scaler_name)
            params = model_def['params']
            print(f'Starting: Model={model_name}, Scaler={scaler_name}')
            tuned_model = tune_model(pipe, params, X_train, Y_train)
            metrics = evaluate_model(model_name, scaler_name, tuned_model, X_test, Y_test)
            results.append(metrics)
            cv_score = tuned_model.best_score_ if hasattr(tuned_model, 'best_score_') else 0
            if cv_score > best_score:
                best_model = tuned_model
                best_score = cv_score
                best_model_filename = f'{model_name}_{scaler_name}_best'
        if best_model:
            save_model(best_model, best_model_filename)
            print(f'Best {model_name} Model Saved: {best_model_filename}.pkl')
            
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='accuracy', ascending=False)
    results_df.to_csv('./reports/model_evaluation.csv', index=True)
    print("Evaluation Results saved to /reports/model_evaluation.csv")
    
    
if __name__ == '__main__':
    train()