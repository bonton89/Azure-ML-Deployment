# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3.6 - AzureML
#     language: python
#     name: python3-azureml
# ---

# +
import json
import numpy as np
import joblib
from azureml.core.model import Model
from azureml.core import Workspace



def init():
    global model
    ws = Workspace.get(name="AzureML01", subscription_id="1b05e210-e039-4537-bc3a-702e1b32d7c9", resource_group= "AzureRG01")
    model_obj = Model(ws, "loan-prediction-model" )
    model_path = model_obj.download(exist_ok = True)
    model = joblib.load(model_path)


def run(data):
    try:
        result = model.predict(data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
# -


