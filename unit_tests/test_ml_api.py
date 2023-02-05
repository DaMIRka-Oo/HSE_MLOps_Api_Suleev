import pytest
from unittest import mock

from api_service.base import fit_model

@mock.patch('api_service.base.log_to_db')
def test_fit_model(mock_sessionmaker: mock.MagicMock,
                   fit_model_data_value_error):
    with pytest.raises(ValueError):
        fit_model(**fit_model_data_value_error)
