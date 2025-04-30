import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from module_model import ModelEvaluation, mlflow_logger


class Models():
    def __init__(self,
                X: pd.DataFrame, 
                y: pd.Series, 
                tag: str, 
                test_size: float = 0.3, 
                shuffle: bool = True, 
                random_state: int = 42):
        """
        : param:
        """

        self.X = X
        self.y = y
        self.random_state = random_state
        self.test_size = test_size
        self.shuffle = shuffle
        self.tag = tag


    def log_regression(self,
                        model: type = LogisticRegression,
                        gridsearch: bool = False,
                            max_iter: int = 5000,
                            param_grid: dict = None,
                            scoring: str = 'f1',
                            cv: int = 5,
                        model_evaluation: bool = True):

        self.gridsearch = gridsearch
        self.model_evaluation= model_evaluation
        self.model = model
        self.evaluator = ModelEvaluation()

        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y,
                                                            random_state=self.random_state,
                                                            test_size=self.test_size, 
                                                            shuffle=self.shuffle)

        if self.gridsearch:
            self.model = self.evaluator.evaluate_with_gridsearch(X_train = X_train,
                                                                  y_train = y_train,
                                                                  base_model = self.model(max_iter = max_iter),
                                                                  param_grid = param_grid,
                                                                  scoring = scoring,
                                                                  cv = cv
                                                                  )
        
        if self.model_evaluation:
            score = self.evaluator.evaluate_model(self.model, X_train, y_train, X_test, y_test, self.tag)
        
        return self

