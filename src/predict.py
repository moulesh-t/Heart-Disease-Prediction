import joblib
import os
import pandas as pd
from config import *
from sklearn.model_selection import train_test_split

def main():
    model_name = input("Enter the model to be used: ")
    model_path = os.path.join('./models', f'{model_name}.pkl')
    
    # Check for model path
    if not os.path.exists(model_path):
        print(F"Model file not found: {model_path}")
        return
    
    # Load Model
    model = joblib.load(model_path)
    print(f"Loaded model: {model_name}")
    
    # Load test data
    data_path = os.path.join('./data','heart.csv')
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    data =pd.read_csv(data_path)
    X = data.drop(['target'],axis=1)
    Y = data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=RANDOM_STATE)
    y_pred = model.predict(X_test)
    
    print("\n Predicted vs actual values:")
    for i in range(len(X_test)):
        print(f'Sample {i+1}: Predicted={y_pred[i]}, Actual={Y_test.iloc[i]}')
        
        
if __name__ == '__main__':
    main()