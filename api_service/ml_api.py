from base import get_model

from flask import Flask, Response, jsonify
from flask_restx import Resource, Api, reqparse

import pandas as pd
import pickle
import os

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
    def post(self):
        args = fit_model_parser.parse_args(strict=True)
        model_type = args.get('model_type')
        model_name = args.get('model_name')
        params = args.get('hyperparams')
        train_data = args.get('train_data')
        train_target = args.get('train_target')

        if params is None:
            params = {}
        model = get_model(model_type, params)

        location = '../models/'
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

        pickle.dump(model, open(f'../models/{model_name}.pkl', 'wb'))

        return Response(status=201)


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

        location = '../models/'
        models = os.listdir(location)

        if f'{model_name}.pkl' not in models:
            raise NameError(f"You must point off existing 'model_name'")

        filename = f'{location}{model_name}.pkl'
        model = pickle.load(open(filename, 'rb'))

        model.set_params(**params)
        model.fit(train_data, train_target)

        pickle.dump(model, open(f'../models/{model_name}.pkl', 'wb'))

        return Response(status=201)

remove_model_parser = reqparse.RequestParser()
remove_model_parser.add_argument('model_names', type=list, location='json',
                                required=True, help='Names of files with model')

@flask_api.route('/remove_model/')
class RemoveModel(Resource):
    @flask_api.expect(remove_model_parser)
    def post(self):
        args = remove_model_parser.parse_args(strict=True)
        model_names = args.get('model_names')

        location = '../models/'
        models = os.listdir(location)

        for model_name in model_names:
            if f'{model_name}.pkl' not in models:
                raise NameError(f"You must point off existing 'model_name'")

            file = f"{model_name}.pkl"
            path = os.path.join(location, file)
            os.remove(path)

        return Response(status=200)


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

        location = '../models/'
        models = os.listdir(location)

        if f'{model_name}.pkl' not in models:
            raise NameError(f"You must point off existing 'model_name'")

        filename = f'{location}{model_name}.pkl'
        model = pickle.load(open(filename, 'rb'))

        predict = model.predict_proba(data)[:, 1]
        if cutoff is not None:
            predict = predict > cutoff
            predict = list(map(int, predict))
        else:
            predict = list(predict)

        return jsonify({"predict": predict})


if __name__ == '__main__':
    flask_app.run(debug=True, host='0.0.0.0', port=5000)



