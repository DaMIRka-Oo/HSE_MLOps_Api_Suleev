import os
import pickle


def get_model(model_type: str, params: dict):
    if model_type == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    elif model_type == 'LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    else:
        raise ValueError('You can work only with '
                         'LogisticRegression and LightGBM')


def fit_model(model_type: str,
              model_name: str,
              params: dict,
              train_data: list,
              train_target: list) -> None:
    if len(train_data) != len(train_target):
        raise ValueError("'train_data' and 'train_target' "
                         "must have the same dimension")

    if params is None:
        params = {}
    model = get_model(model_type, params)

    location = './models/'
    models = os.listdir(location)

    if f'{model_name}.pkl' in models:
        raise NameError(f"Model '{model_name}' already exist!")

    if model_name is None:
        i = 1
        while True:
            if f"model{i}.pkl" in models:
                i += 1
                continue
            break
        model_name = f"model{i}"

    model.fit(train_data, train_target)

    pickle.dump(model, open(f'models/{model_name}.pkl', 'wb'))


def refit_model(model_name: str,
                params: dict,
                train_data: list,
                train_target: list) -> None:
    if len(train_data) != len(train_target):
        raise ValueError("'train_data' and 'train_target' "
                         "must have the same dimension")

    location = './models/'
    models = os.listdir(location)

    if f'{model_name}.pkl' not in models:
        raise NameError("You must point off existing 'model_name'")

    filename = f'{location}{model_name}.pkl'
    model = pickle.load(open(filename, 'rb'))

    if params is not None:
        model.set_params(**params)
    model.fit(train_data, train_target)

    pickle.dump(model, open(f'models/{model_name}.pkl', 'wb'))


def remove_model(model_names: list) -> None:
    location = './models/'
    models = os.listdir(location)

    for model_name in model_names:
        if f'{model_name}.pkl' not in models:
            raise NameError("You must point off existing 'model_name'")

        file = f"{model_name}.pkl"
        path = os.path.join(location, file)
        os.remove(path)


def predict(model_name: str,
            data: list,
            cutoff: float) -> list:
    location = './models/'
    models = os.listdir(location)

    if f'{model_name}.pkl' not in models:
        raise NameError("You must point off existing 'model_name'")

    filename = f'{location}{model_name}.pkl'
    model = pickle.load(open(filename, 'rb'))

    pred = model.predict_proba(data)[:, 1]
    if cutoff is not None:
        pred = pred > cutoff
        pred = list(map(int, pred))
    else:
        pred = list(pred)

    return pred


def show(model_name: str) -> dict:
    location = './models/'
    models = os.listdir(location)
    models.remove('description.txt')

    if model_name == 'All':
        return {"Models": models}

    if f'{model_name}.pkl' not in models:
        raise NameError("You must point off existing 'model_name'")

    filename = f'{location}{model_name}.pkl'
    model = pickle.load(open(filename, 'rb'))

    model_params = model.get_params()
    return {model_name: model_params}