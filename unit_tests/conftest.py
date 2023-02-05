import pytest

@pytest.fixture(scope='session')
def fit_model_data_value_error():
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