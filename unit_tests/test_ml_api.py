import pytest
from unittest import mock
import os

from api_service.base import (
    fit_model,
    refit_model,
    predict,
    show
)

@mock.patch('api_service.base.log_to_db')
def test_fit_model(mock_sessionmaker: mock.MagicMock,
                   fit_model_test_value_error,
                   fit_model_test_name_error):
    with pytest.raises(ValueError):
        fit_model(**fit_model_test_value_error)
    with pytest.raises(NameError):
        fit_model(**fit_model_test_name_error)


@mock.patch('api_service.base.log_to_db')
def test_refit_model(mock_sessionmaker: mock.MagicMock,
                   refit_model_test_value_error,
                   refit_model_test_name_error):
    with pytest.raises(ValueError):
        refit_model(**refit_model_test_value_error)
    with pytest.raises(NameError):
        refit_model(**refit_model_test_name_error)

@mock.patch('api_service.base.log_to_db')
def test_predict(mock_sessionmaker: mock.MagicMock,
                 predict_test,
                 predict_test_name_error):
    with pytest.raises(NameError):
        predict(**predict_test_name_error)

    pred = predict(**predict_test)
    assert len(pred) == 4

@mock.patch('api_service.base.log_to_db')
def test_show(mock_sessionmaker: mock.MagicMock,
             show_test_all,
             show_test_model):
    location = 'api_service/models/'
    models = os.listdir(location)
    models.remove('description.txt')

    model_params = show(**show_test_all)
    assert len(model_params) == len(models)

    model_params = show(**show_test_model)
    model_nm = list(model_params.keys())[0]

    assert model_params[model_nm]['penalty'] == 'l2'
    assert model_params[model_nm]['random_state'] is None
    assert model_params[model_nm]['solver'] == 'lbfgs'


