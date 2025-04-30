# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import functools

# Sklearn
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# XGBoost
from xgboost import XGBClassifier
import xgboost as xgb

# MLFlow
import mlflow
from mlflow.models.signature import infer_signature

# External modules
from module_path import plots_data_path, mlruns_data_path, submission_data_path
from module_graph import graph_confusion_matrix

def mlflow_logger(func):
    """Decorator to automatically start and close an mlflow run"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        #create a new experiment if not in mlruns directory
        mlruns_path = mlruns_data_path()
        mlflow.set_tracking_uri(mlruns_path)
        #print(mlflow.get_artifact_uri())
        experiment_name = 'WIDS2025'

        try:
            exp_id = mlflow.create_experiment(name=experiment_name)
        except Exception as e:
            exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        with mlflow.start_run(experiment_id=exp_id):
            return func(*args, **kwargs)
    return wrapper


class ModelEvaluation:
    """
    Supports the evaluation of classification models (multinomial), collecting the results.
    """
    def __init__(self):
        pass        

    @mlflow_logger
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, tag) -> float:
        """
        :param model: the model to evaluate
        :return: the f1-score
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\nModel evaluation: {type(model).__name__} - {tag}")

        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion matrix ({tag}):\n{cm}")

        graph_confusion_matrix(cm, type(model).__name__, self.tag, labels=sorted(set(y_test)))
        #mlflow.log_artifact(graph_confusion_matrix)

        f1_score = metrics.f1_score(y_test, y_pred)
        mlflow.log_metric("f1_score", f1_score)
        print(f"\nF1_score  : {f1_score:.2f}")

        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("acc", acc)
        print(f"Accuracy  : {acc:.2f}")

        prec = precision_score(y_test, y_pred)
        mlflow.log_metric("prec", prec)
        print(f"Precision : {prec:.2f}")

        rec = recall_score(y_test, y_pred)
        mlflow.log_metric("recall", rec)
        print(f"Recall    : {rec:.2f}")

        #Update model name in MLFlow.
        run_label = f"{type(model).__name__}_{tag}_f1_score={f1_score:.5f}"
        mlflow.set_tag("mlflow.runName", run_label)

    def evaluate_with_gridsearch(self, 
                                 X_train: pd.DataFrame, 
                                 y_train: pd.Series, 
                                 base_model, 
                                 param_grid, 
                                 scoring='f1', 
                                 cv=5, 
                                 save_best_params_path=None):
        """
        Aplica GridSearchCV y evalúa el mejor modelo encontrado.
        :param base_model: el modelo base
        :param param_grid: diccionario con los hiperparámetros a probar
        :param scoring: métrica de evaluación para el grid search
        :param cv: número de folds en la validación cruzada
        :return: el mejor modelo y su f1_score
        """
        print(f"Ejecutando GridSearchCV para {type(base_model).__name__} - {self.tag}")
    
        grid_search = GridSearchCV(
                                    estimator=base_model,
                                    param_grid=param_grid,
                                    cv=cv,
                                    scoring=scoring
                                  )
    
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"\nMejores hiperparámetros para {self.tag}:")
        for param, val in best_params.items():
            print(f"  {param}: {val}")

        #score = self.evaluate_model(best_model)
    
        return best_model
    
class ModelEvaluationXG:
    """
    Supports the evaluation of classification models (xgboost), collecting the results.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, tag: str, test_size: float = 0.3, shuffle: bool = True, random_state: int = 42):
        """
        :param X: the inputs
        :param y: the prediction targets
        :param test_size: the fraction of the data to reserve for testing
        :param shuffle: whether to shuffle the data prior to splitting
        :param random_state: the random seed
        :param tag: target name for logging
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
            random_state=random_state, test_size=test_size, shuffle=shuffle)
        
        self.tag = tag

    @mlflow_logger
    def evaluate_model(self) -> float:
        """
        :param model: the model to evaluate
        :return: the f1-score
        """
        matrix_train_xgb=xgb.DMatrix(self.X_train,label=self.y_train)
        matrix_test_xgb=xgb.DMatrix(self.X_test)  

        param ={
        'max_depth': 4,
        'eta': 0.3,
        'objective':'multi:softmax',
        'tree_method': 'auto',
        'num_class': 3}
        epochs=10

        model_xgb = xgb.train(params=param,dtrain=matrix_train_xgb,num_boost_round=epochs)

        y_pred = model_xgb.predict(matrix_test_xgb)

        f1_score = metrics.f1_score(self.y_test, y_pred)
        print(f"XGBoost model: f1_score={f1_score:.2f}")

        # log parameters and metrics in MLFlow
        mlflow.log_param("Model Type", 'XGBoost_' + self.tag)
        for k, v in param.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("f1_score", f1_score)
        signature = infer_signature(self.X_train, y_pred)
        mlflow.xgboost.log_model(model_xgb, "model", signature=signature)

        #Update model name in MLFlow.
        run_label = f"XGBoost_{self.tag}_f1_score={f1_score:.5f}"
        mlflow.set_tag("mlflow.runName", run_label)

        return f1_score

    
    
class ModelSubmission:

    """
    Supports the submission of a model using mlflow, using a dataset
    """

    def __init__(self, X: pd.DataFrame, version: int=1, threshold: float = 0.5, 
                 adhd_tag: str = "adhd", sex_f_tag: str = "sex_f"
        ):
        """
        :param X: the inputs of the test dataset
        :param version: version of the registered model
        :param threshold: threshold for predicting labels based on predicted probability
        """

        self.X = X
        self.version = version
        self.threshold = threshold
        self.sex_f_tag = sex_f_tag
        self.adhd_tag = adhd_tag

    def load_model(self):
        """
        :return: (tuple) The sklearn models registered in mlflow for sex_f and adhd, respectively
        """
        try:
            mlflow.set_tracking_uri(mlruns_data_path())
            # If model doesn't exist it will raise an exception that will be catched
            # by the to_submission method
            model_sex_f = mlflow.sklearn.load_model(f"models:/Model_{self.sex_f_tag}/{self.version}")
            model_adhd = mlflow.sklearn.load_model(f"models:/Model_{self.adhd_tag}/{self.version}")

            return model_sex_f, model_adhd

        except Exception as e:
            raise Exception(f"Error loading models: {e}")

    def predictions_proba(self):
        """
        :return: (tuple) Predicted probabilities for 1 class (sex_f, adhd)
        """
        
        model_sex_f, model_adhd = self.load_model()
        
        sex_proba = model_sex_f.predict_proba(self.X)
        adhd_proba = model_adhd.predict_proba(self.X)

        return sex_proba[:,1], adhd_proba[:,1]
    
    def predictions_labels(self):
        """
        :return: (tuple) Predicted probabilities for 1 class (sex_f, adhd)
        """
        
        model_sex_f, model_adhd = self.load_model()
        
        sex_labels = model_sex_f.predict(self.X)
        adhd_labels = model_adhd.predict(self.X)

        return sex_labels, adhd_labels
    
    def predictions_labels_from_proba(self):
        """
        :return: (tuple) Array of predicted classes for (sex_f, adhd)
        """

        sex_proba, adhd_proba = self.predictions_proba()

        sex_labels = np.where(sex_proba > self.threshold, 1, 0)
        adhd_labels = np.where(adhd_proba > self.threshold, 1, 0)

        return sex_labels, adhd_labels
    
    def to_submission(self, output_name: str):
        """
        Writes a csv file based on the submission form
        """
        try:
            sex_labels, adhd_labels = self.predictions_labels_from_proba()

            submission = pd.read_excel("../data/SAMPLE_SUBMISSION.xlsx")

            submission["ADHD_Outcome"] = adhd_labels
            submission["Sex_F"] = sex_labels

            submission.to_csv(os.path.join(submission_data_path(), output_name), index=False)
        except Exception as e:
            print(f"Error submitting models: {e}")
  
