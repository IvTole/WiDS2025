import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from module_model import ModelEvaluation, mlflow_logger


class Models():
    '''
    Clase que almacena en cada uno de sus métodos los distintos modelos aplicados al proyecto.
    '''
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
        '''
        Engloba aquellos parametros que comparten en común los distintos modelos.
        : param: X= es el dataframe que contiene los datos a analizar.
        : param: y1 y y2= son los resultados para comparar nuetro modelo (es una lista de tuplas). 
        : param: tag1 y tag2= Son las etiquetas que le daremos a nuestras predicciones.
        : param: test_size= Tamaño de la muestra que será usada para el split (30% por defatult).
        : param: shuffle= Si deseamos barajear los datos antes del split (True por default).
        : param:random_state= Semilla aleatoria (asignamos un valor por default).
        '''

    def log_regression(self,
                       model: type = LogisticRegression,
                       gridsearch: bool = False,
                            param_grid: dict = None,
                            scoring: str = 'f1',
                            cv: int = 5,
                       model_evaluation: bool = False,
                            max_iter: int = 5000,
                            solver: str = 'lbfgs'):
        '''
        Método que contiene al modelo de regresión logistica.
        : param: model= Se llama al tipo de modelo que representa el método en este caso LogisticRegression.
        : param: gridsearch= Valor booleano para activar gridsearchcv.
        : param: param_grid= Diccionario que contiene los valores del gridsearch se ejecturan en una variable aparte en el código.
        : param: scoring= Es el valor que tomará gridseach para hacer las comparativas entre los mejores modelos (f1 por default).
        : param: cv= Número de folds para la validación cruzada (5 por default).
        : param: model_evaluation= Valor booleano para activar la evaluación del modelo.
        : param: max_iter= Número máximo de interaciones si se ejecutar la evaluación del modelo.
        : param: solver: Tipo de solver que usará el modelo para su evaluación.
        : return: Nos dará una tupla con los valores uno para cada una de las dos etiquetas utilizadas.
        '''
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
        '''
        Método que contiene al modelo de regresión logistica.
        : param: model= Se llama al tipo de modelo que representa el método en este caso RandomForestClassifier.
        : param: gridsearch= Valor booleano para activar gridsearchcv.
        : param: param_grid= Diccionario que contiene los valores del gridsearch se ejecturan en una variable aparte en el código.
        : param: scoring= Es el valor que tomará gridseach para hacer las comparativas entre los mejores modelos (f1 por default).
        : param: cv= Número de folds para la validación cruzada (5 por default).
        : param: model_evaluation= Valor booleano para activar la evaluación del modelo.
        : param: n_estimatos= Número total de arboles que conformarán al bosque (1000 por default).  
        : param: criterion= Criterio a usar para realizar los arboles (gini, entropia).
        : param: max_depth= Número que conforma la profundidad del árbol (10 por default).
        : param: random_state: Semilla aleatoria (42 por default).
        : param: bootstrap: Muestreo aleatorio por remplazo (activado por default).
        : return: 
        '''

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
