

def get_model(model_type: str, params: dict):
    if model_type == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    elif model_type == 'LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    else:
        raise ValueError('You can work only with LogisticRegression and LightGBM')
