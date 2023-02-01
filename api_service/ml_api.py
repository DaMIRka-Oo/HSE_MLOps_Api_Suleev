from base import (
    fit_model,
    refit_model,
    remove_model,
    predict,
    show
)

from flask import Flask, jsonify
from flask_restx import Resource, Api, reqparse

flask_app = Flask(__name__)
flask_api = Api(flask_app)

fit_model_parser = reqparse.RequestParser()
fit_model_parser.add_argument('model_type', type=str, location='json',
                              required=True, help='LogisticRegression or LightGBM')
fit_model_parser.add_argument('model_name', type=str, location='json',
                              required=False, help='Name of file with model')
fit_model_parser.add_argument('hyperparams', type=dict, location='json',
                              required=False, help='Model hyperparameters')
fit_model_parser.add_argument('train_data', type=list, location='json',
                              required=True, help='Training data')
fit_model_parser.add_argument('train_target', type=list, location='json',
                              required=True, help='Training target')

@flask_api.route('/fit_model/')
class FitModel(Resource):
    @flask_api.expect(fit_model_parser)
    def put(self):
        args = fit_model_parser.parse_args(strict=True)
        model_type = args.get('model_type')
        model_name = args.get('model_name')
        params = args.get('hyperparams')
        train_data = args.get('train_data')
        train_target = args.get('train_target')

        fit_model(model_type, model_name, params,
                  train_data, train_target)

        response = get_common_response({}, HttpStatus.CREATED)
        return response


refit_model_parser = reqparse.RequestParser()
refit_model_parser.add_argument('model_name', type=str, location='json',
                              required=True, help='Name of file with model')
refit_model_parser.add_argument('hyperparams', type=dict, location='json',
                              required=False, help='Model hyperparameters')
refit_model_parser.add_argument('train_data', type=list, location='json',
                              required=True, help='Training data')
refit_model_parser.add_argument('train_target', type=list, location='json',
                              required=True, help='Training target')

@flask_api.route('/refit_model/')
class RefitModel(Resource):
    @flask_api.expect(refit_model_parser)
    def post(self):
        args = refit_model_parser.parse_args(strict=True)
        model_name = args.get('model_name')
        params = args.get('hyperparams')
        train_data = args.get('train_data')
        train_target = args.get('train_target')

        refit_model(model_name, params,
                  train_data, train_target)

        response = get_common_response({}, HttpStatus.CREATED)
        return response

remove_model_parser = reqparse.RequestParser()
remove_model_parser.add_argument('model_names', type=list, location='json',
                                required=True, help='Names of files with model')

@flask_api.route('/remove_model/')
class RemoveModel(Resource):
    @flask_api.expect(remove_model_parser)
    def delete(self):
        args = remove_model_parser.parse_args(strict=True)
        model_names = args.get('model_names')

        remove_model(model_names)

        response = get_common_response({}, HttpStatus.OK)
        return response


predict_parser = reqparse.RequestParser()
predict_parser.add_argument('model_name', type=str, location='json',
                                  required=True, help='Name of file with model')
predict_parser.add_argument('data', type=list, location='json',
                                  required=True, help='Data')
predict_parser.add_argument('cutoff', type=float, location='json',
                                  required=False, help='Cut-off for predict')

@flask_api.route('/predict/')
class Predict(Resource):
    @flask_api.expect(predict_parser)
    def post(self):
        args = predict_parser.parse_args(strict=True)
        model_name = args.get('model_name')
        data = args.get('data')
        cutoff = args.get('cutoff')

        pred = predict(model_name, data, cutoff)

        response = get_common_response({"predict": pred}, HttpStatus.OK)
        return response


show_parser = reqparse.RequestParser()
show_parser.add_argument('model_name', type=str, location='json',
                            required=True, help='Name of file with model')

@flask_api.route('/show/')
class Show(Resource):
    @flask_api.expect(show_parser)
    def get(self):
        args = show_parser.parse_args(strict=True)
        model_name = args.get('model_name')

        model_params = show(model_name)

        response = get_common_response(model_params, HttpStatus.OK)
        return response

class HttpStatus:
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    NOT_FOUND = 404
    CONFLICT = 409


def get_error_response(exception, status_code):
    construct = {
        'error': exception.__str__(),
        'success': False,
        'result': []
    }
    response = jsonify(construct)
    response.status_code = status_code
    return response

def get_common_response(result, status_code):
    construct = {
        'error': [],
        'success': True,
        'result': result
    }
    response = jsonify(construct)
    response.status_code = status_code
    return response


if __name__ == '__main__':
    flask_app.run(debug=True, host='0.0.0.0', port=5000)



