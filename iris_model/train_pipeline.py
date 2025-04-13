import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from iris_model.config.core import config
from iris_model.pipeline import iris_pipe
from iris_model.processing.data_manager import load_dataset, save_pipeline

from sklearn.preprocessing import LabelEncoder

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name = config.app_config_.training_data_file)
    
    # Features
    X = data[config.model_config_.features].values
    
    # Apply the mapping to the target variable BEFORE the pipeline
    Y = data[config.model_config_.target].values
    
    # For example encoding target feature y
    enc = LabelEncoder()
    label_encoder = enc.fit(Y)
    y = label_encoder.transform(Y)
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,     # predictors
        y,       # target
        test_size = config.model_config_.test_size,
        random_state=config.model_config_.random_state,   # set the random seed here for reproducibility
        stratify=y
    )

    # Pipeline fitting
    iris_pipe.fit(X_train, y_train)
    y_pred = iris_pipe.predict(X_test)

    # Calculate the score/error
    print("Precision:", round(precision_score(y_test, y_pred, average=config.model_config_.evaluation_average), 2))
    print("Recall:", round(recall_score(y_test, y_pred, average=config.model_config_.evaluation_average), 2))
    print("F1 score:", round(f1_score(y_test, y_pred, average=config.model_config_.evaluation_average), 2))

    # persist trained model
    save_pipeline(pipeline_to_persist = iris_pipe)
    
if __name__ == "__main__":
    run_training()