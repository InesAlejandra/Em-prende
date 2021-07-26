#this file must be copied to /Src/api
import sys
from flask import Blueprint, jsonify, request
import pandas as pd
import dill as pickle
import json
#from Src.utils.DataPreparation import *
sys.path.insert(0, '/content/Src/utils/')
import DataPreparation
from DataPreparation import DataPreparation

predict_api = Blueprint('predict_api', __name__)

@predict_api.route('/predict', methods=['POST'])
def apicall():
    try:
        test_json_dump = json.dumps(request.get_json())
        test_df = pd.read_json(test_json_dump, orient='records')
        # Because of request processing Age is being considered as object, but it needs to be float type.
#        test_df['Age'] =    test_df.Age.convert_objects(convert_numeric=True)
        #Getting the RUC separated out
        ruc_ids = test_df['RUC']
    except Exception as e:
        print(':::: Exception occurred while reading json content ::::')
        raise e
 
    if test_df.empty:
        return(bad_request())
    else:
        #Load the saved model
        loaded_model = None
        filename = 'GradientBoostingClassifier_v1.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
#        with open('./Src/ml-model/voting_classifier_v1.pk','rb') as model:
#        with open('./Src/ml-model/GradientBoostingClassifier_v1.sav','rb') as model:
#            loaded_model = pickle.load(model)
        # Before we make any prediction, let's pre-process first.
        data_preparation = DataPreparation()
        test_df = data_preparation.preprocess(test_df)
        test_df.head(10)
        print(f'After pre-process test df — \n {test_df}')
        predictions = loaded_model.predict(test_df)

        prediction_series = list(pd.Series(predictions))

        final_predictions = pd.DataFrame({'RUC': ruc_ids, 'p': prediction_series})

        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200

        return (responses)
