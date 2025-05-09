import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from module_model import ModelEvaluation, mlflow_logger


class Models():
    def __init__(self,
                 X: pd.DataFrame, 
                 y1: pd.Series, y2: pd.Series, 
                 tag1: str, tag2: str, 
                 test_size: float = 0.3, 
                 shuffle: bool = True, 
                 random_state: int = 42):
        self.X = X
        self.y_pairs = [(y1, tag1), (y2, tag2)]
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.evaluator = ModelEvaluation()

    def log_regression(self,
                       model: type = LogisticRegression,
                       gridsearch: bool = False,
                            param_grid: dict = None,
                            scoring: str = 'f1',
                            cv: int = 5,
                       model_evaluation: bool = False,
                            max_iter: int = 5000,
                            solver: str = 'lbfgs'):

        results = []

        for y, tag in self.y_pairs:
            X_train, X_test, y_train, y_test = train_test_split(
                                                                self.X, y,
                                                                random_state=self.random_state,
                                                                test_size=self.test_size,
                                                                shuffle=self.shuffle
                                                               )

            if gridsearch:
                base_model = model(max_iter=max_iter)
                trained_model = self.evaluator.evaluate_with_gridsearch(
                                                                        X_train=X_train,
                                                                        y_train=y_train,
                                                                        base_model=base_model,
                                                                        param_grid=param_grid,
                                                                        tag=tag,
                                                                        scoring=scoring,
                                                                        cv=cv
                                                                      )
            else:
                trained_model = model(solver=solver, max_iter=max_iter)

            if model_evaluation:
                self.evaluator.evaluate_model(trained_model, X_train, y_train, X_test, y_test, tag)

            results.append(trained_model)

        return tuple(results)
    


    def random_forest(self,
                  model: type = RandomForestClassifier,
                    gridsearch: bool = False,
                    param_grid: dict = None,
                    scoring: str = 'f1',
                    cv: int = 5,
                  model_evaluation: bool = True,
                    n_estimators: int = 1000,
                    criterion: str = "gini",
                    max_depth: int = 10,
                    random_state: int = 42,
                    bootstrap: bool = True):

        results = []

        for y, tag in self.y_pairs:
            X_train, X_test, y_train, y_test = train_test_split(
                                                                self.X, y,
                                                                random_state=self.random_state,
                                                                test_size=self.test_size,
                                                                shuffle=self.shuffle
                                                               )

            if gridsearch:
                base_model = model()
                trained_model = self.evaluator.evaluate_with_gridsearch(
                                                                        X_train=X_train,
                                                                        y_train=y_train,
                                                                        base_model=base_model,
                                                                        param_grid=param_grid,
                                                                        tag=tag,
                                                                        scoring=scoring,
                                                                        cv=cv
                                                                       )
            else:
                trained_model = model(
                                        n_estimators=n_estimators,
                                        criterion=criterion,
                                        max_depth=max_depth,
                                        random_state=random_state,
                                        bootstrap=bootstrap
                                    )
                trained_model.fit(X_train, y_train)

            if model_evaluation:
                self.evaluator.evaluate_model(trained_model, X_train, y_train, X_test, y_test, tag)

            results.append(trained_model)

        return tuple(results)
