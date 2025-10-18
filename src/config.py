TEST_SIZE = 0.2
RANDOM_STATE = 42

#Logistic Regression Parameters
LG_PARAMS = [{
    'model__C': [0.001, 0.01, 0.1, 1, 10],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear']
    },
    {
    'model__C': [0.001, 0.01, 0.1, 1, 10],
    'model__penalty': ['elasticnet','l1','l2',None],
    'model__solver': ['saga']
    }]

RF_PARAMS = {
    'model__n_estimators': [100,200,300],
    'model__max_depth': [50,100,200],
    'model__criterion': ['gini','entropy','log_loss'],
    'model__min_samples_split': [2,5,10]
}

SVC_PARAMS = {
    'model__C': [0.001, 0.01, 0.1, 1],
    'model__kernel': ['linear','poly','rbf','sigmoid']
}