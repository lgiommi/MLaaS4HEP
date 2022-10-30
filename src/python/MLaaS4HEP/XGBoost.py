import xgboost as xgb

def model():
    params = {
        'objective':'binary:logistic',
        'max_depth': 7,
        'subsample': 0.9,
        'alpha': 10,
        'learning_rate': 0.0001,
        'n_estimators': 1500
    }
    xgb_model = xgb.XGBClassifier(**params)
    return xgb_model
