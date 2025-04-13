import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parent.parent  # Go up two levels to the project root
sys.path.append(str(parent))

import pandas as pd
import numpy as np
from typing import Dict, Any

# Now you should be able to import predict.py
from iris_model.predict import make_prediction, _version

def test_make_prediction_valid_input():
    """Test prediction with valid input data."""
    input_data = {'SepalLength': [5.1], 'SepalWidth': [3.5], 'PetalLength': [1.4], 'PetalWidth': [0.2]}
    result = make_prediction(input_data=pd.DataFrame(input_data))  # Pass the DataFrame

    assert "predictions" in result
    assert isinstance(result["predictions"], np.ndarray)
    assert result["predictions"].shape == (1,)
    assert result["errors"] is None
    assert result["version"] == _version

