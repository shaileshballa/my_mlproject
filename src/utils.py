#functionalities all can be done over here
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import dill
import joblib
import pandas as pd
from src.exception import CustomException
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        print(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)