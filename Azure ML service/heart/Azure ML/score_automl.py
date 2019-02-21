
import pickle
import json
import numpy
import azureml.train.automl
import joblib
from azureml.core.model import Model

def init():
    global model
    # note here "lr_model.pickle" is the name of the model registered under
    model_path = Model.get_model_path("automl_model.joblib")
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        data = numpy.array(data)
        result = model.predict(data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
