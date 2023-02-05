import pytest
import os

@pytest.fixture(scope='session')
def fit_model_test_value_error():
    return {
        "model_type": "LogisticRegression",
        "model_name": None,
        "params": {},
        "train_data": [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
            [6.7, 3.1, 4.4, 1.4],
            [5.6, 3.0, 4.5, 1.5],
            [5.8, 2.7, 4.1, 1.0]
        ],
        "train_target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }

@pytest.fixture()
def fit_model_test_name_error():
    location = 'api_service/models/'
    models = os.listdir(location)
    models.remove('description.txt')

    #Убираем расширение
    model = models[0].split(".")[0]

    return {
        "model_type": "LogisticRegression",
        "model_name": model,
        "params": {},
        "train_data": [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
            [6.7, 3.1, 4.4, 1.4],
            [5.6, 3.0, 4.5, 1.5],
            [5.8, 2.7, 4.1, 1.0],
            [6.2, 2.2, 4.5, 1.5],
            [5.6, 2.5, 3.9, 1.1]
        ],
        "train_target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }

@pytest.fixture()
def refit_model_test_value_error():
    location = 'api_service/models/'
    models = os.listdir(location)
    models.remove('description.txt')

    # Убираем расширение
    model = models[0].split(".")[0]

    return {
        "model_name": model,
        "params": {},
        "train_data": [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
            [6.7, 3.1, 4.4, 1.4],
            [5.6, 3.0, 4.5, 1.5],
            [5.8, 2.7, 4.1, 1.0]
        ],
        "train_target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }

@pytest.fixture(scope='session')
def refit_model_test_name_error():
    model = "unexisted_model"

    return {
        "model_name": model,
        "params": {},
        "train_data": [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
            [6.7, 3.1, 4.4, 1.4],
            [5.6, 3.0, 4.5, 1.5],
            [5.8, 2.7, 4.1, 1.0],
            [6.2, 2.2, 4.5, 1.5],
            [5.6, 2.5, 3.9, 1.1]
        ],
        "train_target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }


@pytest.fixture()
def predict_test():
    location = 'api_service/models/'
    models = os.listdir(location)
    models.remove('description.txt')

    # Убираем расширение
    model = models[0].split(".")[0]

    return {
        "model_name": model,
        "data": [
            [5.4, 3.7, 1.5],
           [4.8, 3.4, 1.6],
           [5.9, 3.2, 4.8],
           [6.1, 2.8, 4.0]
        ],
        "cutoff": 0.2
    }

@pytest.fixture(scope='session')
def predict_test_name_error():
    model = "unexisted_model"

    return {
        "model_name": model,
        "data": [
            [5.4, 3.7, 1.5, 0.2],
           [4.8, 3.4, 1.6, 0.2],
           [5.9, 3.2, 4.8, 1.8],
           [6.1, 2.8, 4.0 , 1.3]
        ],
        "cutoff": 0.2
    }

@pytest.fixture(scope='session')
def show_test_all():
    return {
        "model_name": "All"
    }

@pytest.fixture()
def show_test_model():
    location = 'api_service/models/'
    models = os.listdir(location)
    models.remove('description.txt')

    # Убираем расширение
    model = models[0].split(".")[0]

    return {
        "model_name": model
    }
