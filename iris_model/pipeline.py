import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from iris_model.config.core import config
from iris_model.processing.features import Mapper

iris_pipe = Pipeline([
    
    # Classifier
    ('model_dt', DecisionTreeClassifier(max_depth = config.model_config_.max_depth,
                                        criterion = 'entropy',
                                      random_state = config.model_config_.random_state))
    
    ])
