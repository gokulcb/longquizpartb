import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from iris_model import __version__ as _version
from iris_model.config.core import config
from iris_model.processing.data_manager import load_pipeline
from iris_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
iris_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = iris_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        print(results)

    return results



if __name__ == "__main__":

    data_in = {'SepalLength': [5.9], 'SepalWidth': [3], 'PetalLength': [5.1], 'PetalWidth': [1.8]}
    results = make_prediction(input_data = data_in)
