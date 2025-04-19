from typing import Any, List, Optional, Union
import datetime

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class DataInputSchema(BaseModel):
    SepalLength: Optional[float]
    SepalWidth: Optional[float]
    PetalLength: Optional[float]
    PetalWidth: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
